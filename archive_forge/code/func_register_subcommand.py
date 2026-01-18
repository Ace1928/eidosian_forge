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
@staticmethod
def register_subcommand(parser: ArgumentParser):
    add_new_model_like_parser = parser.add_parser('add-new-model-like')
    add_new_model_like_parser.add_argument('--config_file', type=str, help='A file with all the information for this model creation.')
    add_new_model_like_parser.add_argument('--path_to_repo', type=str, help='When not using an editable install, the path to the Transformers repo.')
    add_new_model_like_parser.set_defaults(func=add_new_model_like_command_factory)