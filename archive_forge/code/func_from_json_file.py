import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import yaml
from ...utils import ComputeEnvironment, DistributedType, SageMakerDistributedType
from ...utils.constants import SAGEMAKER_PYTHON_VERSION, SAGEMAKER_PYTORCH_VERSION, SAGEMAKER_TRANSFORMERS_VERSION
@classmethod
def from_json_file(cls, json_file=None):
    json_file = default_json_config_file if json_file is None else json_file
    with open(json_file, encoding='utf-8') as f:
        config_dict = json.load(f)
    if 'compute_environment' not in config_dict:
        config_dict['compute_environment'] = ComputeEnvironment.LOCAL_MACHINE
    if 'mixed_precision' not in config_dict:
        config_dict['mixed_precision'] = 'fp16' if 'fp16' in config_dict and config_dict['fp16'] else None
    if 'fp16' in config_dict:
        del config_dict['fp16']
    if 'dynamo_backend' in config_dict:
        dynamo_backend = config_dict.pop('dynamo_backend')
        config_dict['dynamo_config'] = {} if dynamo_backend == 'NO' else {'dynamo_backend': dynamo_backend}
    if 'use_cpu' not in config_dict:
        config_dict['use_cpu'] = False
    if 'debug' not in config_dict:
        config_dict['debug'] = False
    extra_keys = sorted(set(config_dict.keys()) - set(cls.__dataclass_fields__.keys()))
    if len(extra_keys) > 0:
        raise ValueError(f'The config file at {json_file} had unknown keys ({extra_keys}), please try upgrading your `accelerate` version or fix (and potentially remove) these keys from your config file.')
    return cls(**config_dict)