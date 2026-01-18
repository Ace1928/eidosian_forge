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
def metadata_save(local_path: Union[str, Path], data: Dict) -> None:
    """
    Save the metadata dict in the upper YAML part Trying to preserve newlines as
    in the existing file. Docs about open() with newline="" parameter:
    https://docs.python.org/3/library/functions.html?highlight=open#open Does
    not work with "^M" linebreaks, which are replaced by 

    """
    line_break = '\n'
    content = ''
    if os.path.exists(local_path):
        with open(local_path, 'r', newline='', encoding='utf8') as readme:
            content = readme.read()
            if isinstance(readme.newlines, tuple):
                line_break = readme.newlines[0]
            elif isinstance(readme.newlines, str):
                line_break = readme.newlines
    with open(local_path, 'w', newline='', encoding='utf8') as readme:
        data_yaml = yaml_dump(data, sort_keys=False, line_break=line_break)
        match = REGEX_YAML_BLOCK.search(content)
        if match:
            output = content[:match.start()] + f'---{line_break}{data_yaml}---{line_break}' + content[match.end():]
        else:
            output = f'---{line_break}{data_yaml}---{line_break}{content}'
        readme.write(output)
        readme.close()