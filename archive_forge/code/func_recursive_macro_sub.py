import asyncio
import json
import logging
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast
import click
import wandb
import wandb.docker as docker
from wandb import util
from wandb.apis.internal import Api
from wandb.errors import CommError
from wandb.sdk.launch.errors import LaunchError
from wandb.sdk.launch.git_reference import GitReference
from wandb.sdk.launch.wandb_reference import WandbReference
from wandb.sdk.wandb_config import Config
from .builder.templates._wandb_bootstrap import (
def recursive_macro_sub(source: Any, sub_dict: Dict[str, Optional[str]]) -> Any:
    """Recursively substitute macros in a parsed JSON or YAML blob.

    Macros occur in strings at leaves of the blob in the ${macro} format.
    The macro names are substituted with their values from the given dictionary.
    If a macro is not found in the dictionary, it is left unchanged.

    Arguments:
        source: The JSON or YAML blob to substitute macros in.
        sub_dict: A dictionary mapping macro names to their values.

    Returns:
        The blob with the macros substituted.
    """
    if isinstance(source, str):
        return macro_sub(source, sub_dict)
    elif isinstance(source, list):
        return [recursive_macro_sub(item, sub_dict) for item in source]
    elif isinstance(source, dict):
        return {key: recursive_macro_sub(value, sub_dict) for key, value in source.items()}
    else:
        return source