import copy
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
import yaml
from huggingface_hub import model_info
from huggingface_hub.utils import HFValidationError
from . import __version__
from .models.auto.modeling_auto import (
from .training_args import ParallelMode
from .utils import (
def create_model_index(self, metric_mapping):
    model_index = {'name': self.model_name}
    dataset_names = _listify(self.dataset)
    dataset_tags = _listify(self.dataset_tags)
    dataset_args = _listify(self.dataset_args)
    dataset_metadata = _listify(self.dataset_metadata)
    if len(dataset_args) < len(dataset_tags):
        dataset_args = dataset_args + [None] * (len(dataset_tags) - len(dataset_args))
    dataset_mapping = dict(zip(dataset_tags, dataset_names))
    dataset_arg_mapping = dict(zip(dataset_tags, dataset_args))
    dataset_metadata_mapping = dict(zip(dataset_tags, dataset_metadata))
    task_mapping = {task: TASK_TAG_TO_NAME_MAPPING[task] for task in _listify(self.tasks) if task in TASK_TAG_TO_NAME_MAPPING}
    model_index['results'] = []
    if len(task_mapping) == 0 and len(dataset_mapping) == 0:
        return [model_index]
    if len(task_mapping) == 0:
        task_mapping = {None: None}
    if len(dataset_mapping) == 0:
        dataset_mapping = {None: None}
    all_possibilities = [(task_tag, ds_tag) for task_tag in task_mapping for ds_tag in dataset_mapping]
    for task_tag, ds_tag in all_possibilities:
        result = {}
        if task_tag is not None:
            result['task'] = {'name': task_mapping[task_tag], 'type': task_tag}
        if ds_tag is not None:
            metadata = dataset_metadata_mapping.get(ds_tag, {})
            result['dataset'] = {'name': dataset_mapping[ds_tag], 'type': ds_tag, **metadata}
            if dataset_arg_mapping[ds_tag] is not None:
                result['dataset']['args'] = dataset_arg_mapping[ds_tag]
        if len(metric_mapping) > 0:
            result['metrics'] = []
            for metric_tag, metric_name in metric_mapping.items():
                result['metrics'].append({'name': metric_name, 'type': metric_tag, 'value': self.eval_results[metric_name]})
        if 'task' in result and 'dataset' in result and ('metrics' in result):
            model_index['results'].append(result)
        else:
            logger.info(f'Dropping the following result as it does not have all the necessary fields:\n{result}')
    return [model_index]