import json
import os
import re
import shutil
import sys
import tempfile
import traceback
import warnings
from concurrent import futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
import huggingface_hub
import requests
from huggingface_hub import (
from huggingface_hub.file_download import REGEX_COMMIT_HASH, http_get
from huggingface_hub.utils import (
from huggingface_hub.utils._deprecation import _deprecate_method
from requests.exceptions import HTTPError
from . import __version__, logging
from .generic import working_or_temp_dir
from .import_utils import (
from .logging import tqdm
def send_example_telemetry(example_name, *example_args, framework='pytorch'):
    """
    Sends telemetry that helps tracking the examples use.

    Args:
        example_name (`str`): The name of the example.
        *example_args (dataclasses or `argparse.ArgumentParser`): The arguments to the script. This function will only
            try to extract the model and dataset name from those. Nothing else is tracked.
        framework (`str`, *optional*, defaults to `"pytorch"`): The framework for the example.
    """
    if is_offline_mode():
        return
    data = {'example': example_name, 'framework': framework}
    for args in example_args:
        args_as_dict = {k: v for k, v in args.__dict__.items() if not k.startswith('_') and v is not None}
        if 'model_name_or_path' in args_as_dict:
            model_name = args_as_dict['model_name_or_path']
            if not os.path.isdir(model_name):
                data['model_name'] = args_as_dict['model_name_or_path']
        if 'dataset_name' in args_as_dict:
            data['dataset_name'] = args_as_dict['dataset_name']
        elif 'task_name' in args_as_dict:
            script_name = example_name.replace('tf_', '').replace('flax_', '').replace('run_', '')
            script_name = script_name.replace('_no_trainer', '')
            data['dataset_name'] = f'{script_name}-{args_as_dict['task_name']}'
    send_telemetry(topic='examples', library_name='transformers', library_version=__version__, user_agent=http_user_agent(data))