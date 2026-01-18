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
def macro_sub(original: str, sub_dict: Dict[str, Optional[str]]) -> str:
    """Substitute macros in a string.

    Macros occur in the string in the ${macro} format. The macro names are
    substituted with their values from the given dictionary. If a macro
    is not found in the dictionary, it is left unchanged.

    Args:
        original: The string to substitute macros in.
        sub_dict: A dictionary mapping macro names to their values.

    Returns:
        The string with the macros substituted.
    """
    return MACRO_REGEX.sub(lambda match: str(sub_dict.get(match.group(1), match.group(0))), original)