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
def synupdate(self, groupid: str, skipinitial: bool=False, *terms: List[str]):
    """
        Updates a synonym group.
        The command is used to create or update a synonym group with
        additional terms.
        Only documents which were indexed after the update will be affected.

        Parameters:

        groupid :
            Synonym group id.
        skipinitial : bool
            If set to true, we do not scan and index.
        terms :
            The terms.

        For more information see `FT.SYNUPDATE <https://redis.io/commands/ft.synupdate>`_.
        """
    cmd = [SYNUPDATE_CMD, self.index_name, groupid]
    if skipinitial:
        cmd.extend(['SKIPINITIALSCAN'])
    cmd.extend(terms)
    return self.execute_command(*cmd)