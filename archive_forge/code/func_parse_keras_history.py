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
def parse_keras_history(logs):
    """
    Parse the `logs` of either a `keras.History` object returned by `model.fit()` or an accumulated logs `dict`
    passed to the `PushToHubCallback`. Returns lines and logs compatible with those returned by `parse_log_history`.
    """
    if hasattr(logs, 'history'):
        if not hasattr(logs, 'epoch'):
            return (None, [], {})
        logs.history['epoch'] = logs.epoch
        logs = logs.history
    else:
        logs = {log_key: [single_dict[log_key] for single_dict in logs] for log_key in logs[0]}
    lines = []
    for i in range(len(logs['epoch'])):
        epoch_dict = {log_key: log_value_list[i] for log_key, log_value_list in logs.items()}
        values = {}
        for k, v in epoch_dict.items():
            if k.startswith('val_'):
                k = 'validation_' + k[4:]
            elif k != 'epoch':
                k = 'train_' + k
            splits = k.split('_')
            name = ' '.join([part.capitalize() for part in splits])
            values[name] = v
        lines.append(values)
    eval_results = lines[-1]
    return (logs, lines, eval_results)