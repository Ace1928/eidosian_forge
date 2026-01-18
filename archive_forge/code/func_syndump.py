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
def syndump(self):
    """
        Dumps the contents of a synonym group.

        The command is used to dump the synonyms data structure.
        Returns a list of synonym terms and their synonym group ids.

        For more information see `FT.SYNDUMP <https://redis.io/commands/ft.syndump>`_.
        """
    res = self.execute_command(SYNDUMP_CMD, self.index_name)
    return self._parse_results(SYNDUMP_CMD, res)