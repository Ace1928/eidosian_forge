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
def dict_add(self, name: str, *terms: List[str]):
    """Adds terms to a dictionary.

        ### Parameters

        - **name**: Dictionary name.
        - **terms**: List of items for adding to the dictionary.

        For more information see `FT.DICTADD <https://redis.io/commands/ft.dictadd>`_.
        """
    cmd = [DICT_ADD_CMD, name]
    cmd.extend(terms)
    return self.execute_command(*cmd)