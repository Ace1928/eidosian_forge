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
def define_sagemaker_information():
    try:
        instance_data = requests.get(os.environ['ECS_CONTAINER_METADATA_URI']).json()
        dlc_container_used = instance_data['Image']
        dlc_tag = instance_data['Image'].split(':')[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None
    sagemaker_params = json.loads(os.getenv('SM_FRAMEWORK_PARAMS', '{}'))
    runs_distributed_training = True if 'sagemaker_distributed_dataparallel_enabled' in sagemaker_params else False
    account_id = os.getenv('TRAINING_JOB_ARN').split(':')[4] if 'TRAINING_JOB_ARN' in os.environ else None
    sagemaker_object = {'sm_framework': os.getenv('SM_FRAMEWORK_MODULE', None), 'sm_region': os.getenv('AWS_REGION', None), 'sm_number_gpu': os.getenv('SM_NUM_GPUS', 0), 'sm_number_cpu': os.getenv('SM_NUM_CPUS', 0), 'sm_distributed_training': runs_distributed_training, 'sm_deep_learning_container': dlc_container_used, 'sm_deep_learning_container_tag': dlc_tag, 'sm_account_id': account_id}
    return sagemaker_object