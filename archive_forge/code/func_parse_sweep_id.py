import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def parse_sweep_id(parts_dict: dict) -> Optional[str]:
    """In place parse sweep path from parts dict.

    Arguments:
        parts_dict (dict): dict(entity=,project=,name=).  Modifies dict inplace.

    Returns:
        None or str if there is an error
    """
    entity = None
    project = None
    sweep_id = parts_dict.get('name')
    if not isinstance(sweep_id, str):
        return 'Expected string sweep_id'
    sweep_split = sweep_id.split('/')
    if len(sweep_split) == 1:
        pass
    elif len(sweep_split) == 2:
        split_project, sweep_id = sweep_split
        project = split_project or project
    elif len(sweep_split) == 3:
        split_entity, split_project, sweep_id = sweep_split
        project = split_project or project
        entity = split_entity or entity
    else:
        return 'Expected sweep_id in form of sweep, project/sweep, or entity/project/sweep'
    parts_dict.update(dict(name=sweep_id, project=project, entity=entity))
    return None