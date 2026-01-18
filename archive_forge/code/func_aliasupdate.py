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
def aliasupdate(self, alias: str):
    """
        Updates an alias - will fail if alias does not already exist

        ### Parameters

        - **alias**: Name of the alias to create

        For more information see `FT.ALIASUPDATE <https://redis.io/commands/ft.aliasupdate>`_.
        """
    return self.execute_command(ALIAS_UPDATE_CMD, alias, self.index_name)