import json
import logging
from datetime import datetime
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple
import click
import yaml
import ray._private.services as services
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.state import (
from ray.util.state.common import (
from ray.util.state.exception import RayStateApiException
from ray.util.annotations import PublicAPI
def format_list_api_output(state_data: List[StateSchema], *, schema: StateSchema, format: AvailableFormat=AvailableFormat.DEFAULT, detail: bool=False) -> str:
    if len(state_data) == 0:
        return 'No resource in the cluster'
    state_data = [state.asdict() for state in state_data]
    return output_with_format(state_data, schema=schema, format=format, detail=detail)