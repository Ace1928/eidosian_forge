import argparse
import collections
import json
from pathlib import Path
import requests
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
from transformers.utils import logging
def flatten_yaml_as_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)