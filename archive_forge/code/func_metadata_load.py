import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union
import requests
import yaml
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import upload_file
from huggingface_hub.repocard_data import (
from huggingface_hub.utils import get_session, is_jinja_available, yaml_dump
from .constants import REPOCARD_NAME
from .utils import EntryNotFoundError, SoftTemporaryDirectory, logging, validate_hf_hub_args
def metadata_load(local_path: Union[str, Path]) -> Optional[Dict]:
    content = Path(local_path).read_text()
    match = REGEX_YAML_BLOCK.search(content)
    if match:
        yaml_block = match.group(2)
        data = yaml.safe_load(yaml_block)
        if data is None or isinstance(data, dict):
            return data
        raise ValueError('repo card metadata block should be a dict')
    else:
        return None