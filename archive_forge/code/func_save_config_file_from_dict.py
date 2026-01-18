import json
import logging
import os
from typing import Any, Dict, Optional
import yaml
import wandb
from wandb.errors import Error
from wandb.util import load_yaml
from . import filesystem
def save_config_file_from_dict(config_filename, config_dict):
    s = b'wandb_version: 1'
    if config_dict:
        s += b'\n\n' + yaml.dump(config_dict, Dumper=yaml.SafeDumper, default_flow_style=False, allow_unicode=True, encoding='utf-8', sort_keys=False)
    data = s.decode('utf-8')
    filesystem.mkdir_exists_ok(os.path.dirname(config_filename))
    with open(config_filename, 'w') as conf_file:
        conf_file.write(data)