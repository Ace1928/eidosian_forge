import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
import psutil
import torch
from accelerate.commands.config import default_config_file, load_config_from_file
from accelerate.commands.config.config_args import SageMakerConfig
from accelerate.commands.config.config_utils import DYNAMO_BACKENDS
from accelerate.commands.utils import CustomArgumentParser
from accelerate.state import get_int_from_env
from accelerate.utils import (
from accelerate.utils.constants import DEEPSPEED_MULTINODE_LAUNCHERS, TORCH_DYNAMO_MODES
def sagemaker_launcher(sagemaker_config: SageMakerConfig, args):
    if not is_sagemaker_available():
        raise ImportError('Please install sagemaker to be able to launch training on Amazon SageMaker with `pip install accelerate[sagemaker]`')
    if args.module or args.no_python:
        raise ValueError('SageMaker requires a python training script file and cannot be used with --module or --no_python')
    from sagemaker.huggingface import HuggingFace
    args, sagemaker_inputs = prepare_sagemager_args_inputs(sagemaker_config, args)
    huggingface_estimator = HuggingFace(**args)
    huggingface_estimator.fit(inputs=sagemaker_inputs)
    print(f'You can find your model data at: {huggingface_estimator.model_data}')