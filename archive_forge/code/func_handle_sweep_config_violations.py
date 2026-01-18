import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def handle_sweep_config_violations(warnings: List[str]) -> None:
    """Echo sweep config schema violation warnings from Gorilla to the terminal.

    Parameters
    ----------
    warnings: list of str
        The warnings to render.
    """
    warning = sweep_config_err_text_from_jsonschema_violations(warnings)
    if len(warnings) > 0:
        wandb.termwarn(warning)