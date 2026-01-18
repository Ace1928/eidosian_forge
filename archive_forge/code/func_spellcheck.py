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
def spellcheck(self, query, distance=None, include=None, exclude=None):
    """
        Issue a spellcheck query

        ### Parameters

        **query**: search query.
        **distance***: the maximal Levenshtein distance for spelling
                       suggestions (default: 1, max: 4).
        **include**: specifies an inclusion custom dictionary.
        **exclude**: specifies an exclusion custom dictionary.

        For more information see `FT.SPELLCHECK <https://redis.io/commands/ft.spellcheck>`_.
        """
    cmd = [SPELLCHECK_CMD, self.index_name, query]
    if distance:
        cmd.extend(['DISTANCE', distance])
    if include:
        cmd.extend(['TERMS', 'INCLUDE', include])
    if exclude:
        cmd.extend(['TERMS', 'EXCLUDE', exclude])
    res = self.execute_command(*cmd)
    return self._parse_results(SPELLCHECK_CMD, res)