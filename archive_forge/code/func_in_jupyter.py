import logging
import sys
import warnings
from typing import Optional
import wandb
def in_jupyter() -> bool:
    return _get_python_type() == 'jupyter'