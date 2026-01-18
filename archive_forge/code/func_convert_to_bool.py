import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def convert_to_bool(x: str) -> bool:
    """
    Converts a string to a bool.
    """
    if x.lower() in ['1', 'y', 'yes', 'true']:
        return True
    if x.lower() in ['0', 'n', 'no', 'false']:
        return False
    raise ValueError(f'{x} is not a value that can be converted to a bool.')