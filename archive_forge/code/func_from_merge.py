import copy
import dataclasses
import json
import os
import posixpath
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union
import fsspec
from huggingface_hub import DatasetCard, DatasetCardData
from . import config
from .features import Features, Value
from .splits import SplitDict
from .tasks import TaskTemplate, task_template_from_dict
from .utils import Version
from .utils.logging import get_logger
from .utils.py_utils import asdict, unique_values
@classmethod
def from_merge(cls, dataset_infos: List['DatasetInfo']):
    dataset_infos = [dset_info.copy() for dset_info in dataset_infos if dset_info is not None]
    if len(dataset_infos) > 0 and all((dataset_infos[0] == dset_info for dset_info in dataset_infos)):
        return dataset_infos[0]
    description = '\n\n'.join(unique_values((info.description for info in dataset_infos))).strip()
    citation = '\n\n'.join(unique_values((info.citation for info in dataset_infos))).strip()
    homepage = '\n\n'.join(unique_values((info.homepage for info in dataset_infos))).strip()
    license = '\n\n'.join(unique_values((info.license for info in dataset_infos))).strip()
    features = None
    supervised_keys = None
    task_templates = None
    all_task_templates = [info.task_templates for info in dataset_infos if info.task_templates is not None]
    if len(all_task_templates) > 1:
        task_templates = list(set(all_task_templates[0]).intersection(*all_task_templates[1:]))
    elif len(all_task_templates):
        task_templates = list(set(all_task_templates[0]))
    task_templates = task_templates if task_templates else None
    return cls(description=description, citation=citation, homepage=homepage, license=license, features=features, supervised_keys=supervised_keys, task_templates=task_templates)