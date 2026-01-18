from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def get_all_arguments(orig_args, orig_kwargs):
    all_args = []
    for i, schema in enumerate(args_schema):
        if schema.name in orig_kwargs:
            all_args.append(orig_kwargs[schema.name])
        elif not schema.kwarg_only and i < len(orig_args):
            all_args.append(orig_args[i])
        else:
            all_args.append(schema.default_value)
    return all_args