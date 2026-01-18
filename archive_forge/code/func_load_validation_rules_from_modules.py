import asyncio
import json
import pathlib
import re
import logging
from typing import (
import inspect
from inspect import signature, iscoroutinefunction
from collections.abc import Mapping, Iterable
from enum import Enum
import importlib
import os
import aiofiles
from regex import W
import asyncio
import types
import importlib.util
def load_validation_rules_from_modules(module_paths: List[str]) -> ValidationRules:
    """
    Dynamically loads validation rules from the specified module paths. Validation rules are assumed to be
    functions starting with 'is_' in the module's namespace. This function iterates over each module, inspecting
    its attributes to find validation functions, and then compiles a dictionary of these rules for use in validation.

    Args:
        module_paths (List[str]): A list of paths to modules containing validation rules.

    Returns:
        ValidationRules: A dictionary mapping rule names to their corresponding asynchronous validation functions.
    """
    rules = {}
    for path in module_paths:
        module_name = os.path.splitext(os.path.basename(path))[0]
        try:
            module = loaded_modules[module_name]
            for attr in dir(module):
                if attr.startswith('is_'):
                    rules[attr] = getattr(module, attr)
        except KeyError:
            logging.warning(f'Module {module_name} not found in loaded_modules. Continuing without loading its rules.')
    return rules