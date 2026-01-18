import dataclasses
import os
from typing import Any, List
import torch
from .utils import print_once
def should_print_missing():
    return os.environ.get('TORCHDYNAMO_PRINT_MISSING') == '1'