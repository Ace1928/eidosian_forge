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
def dict_dump(self, name: str):
    """Dumps all terms in the given dictionary.

        ### Parameters

        - **name**: Dictionary name.

        For more information see `FT.DICTDUMP <https://redis.io/commands/ft.dictdump>`_.
        """
    cmd = [DICT_DUMP_CMD, name]
    return self.execute_command(*cmd)