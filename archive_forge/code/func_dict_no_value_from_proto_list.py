import json
import logging
import os
from typing import Any, Dict, Optional
import yaml
import wandb
from wandb.errors import Error
from wandb.util import load_yaml
from . import filesystem
def dict_no_value_from_proto_list(obj_list):
    d = dict()
    for item in obj_list:
        possible_dict = json.loads(item.value_json)
        if not isinstance(possible_dict, dict) or 'value' not in possible_dict:
            continue
        d[item.key] = possible_dict['value']
    return d