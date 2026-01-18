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
def sugadd(self, key, *suggestions, **kwargs):
    """
        Add suggestion terms to the AutoCompleter engine. Each suggestion has
        a score and string.
        If kwargs["increment"] is true and the terms are already in the
        server's dictionary, we increment their scores.

        For more information see `FT.SUGADD <https://redis.io/commands/ft.sugadd/>`_.
        """
    pipe = self.pipeline(transaction=False)
    for sug in suggestions:
        args = [SUGADD_COMMAND, key, sug.string, sug.score]
        if kwargs.get('increment'):
            args.append('INCR')
        if sug.payload:
            args.append('PAYLOAD')
            args.append(sug.payload)
        pipe.execute_command(*args)
    return pipe.execute()[-1]