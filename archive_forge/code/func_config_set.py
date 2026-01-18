import itertools
import time
from typing import Dict, List, Optional, Union
from redis.client import Pipeline
from redis.utils import deprecated_function
from ..helpers import get_protocol_version, parse_to_dict
from ._util import to_string
from .aggregation import AggregateRequest, AggregateResult, Cursor
from .document import Document
from .query import Query
from .result import Result
from .suggestion import SuggestionParser
def config_set(self, option: str, value: str) -> bool:
    """Set runtime configuration option.

        ### Parameters

        - **option**: the name of the configuration option.
        - **value**: a value for the configuration option.

        For more information see `FT.CONFIG SET <https://redis.io/commands/ft.config-set>`_.
        """
    cmd = [CONFIG_CMD, 'SET', option, value]
    raw = self.execute_command(*cmd)
    return raw == 'OK'