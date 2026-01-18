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
def load_document(self, id):
    """
        Load a single document by id
        """
    fields = self.client.hgetall(id)
    f2 = {to_string(k): to_string(v) for k, v in fields.items()}
    fields = f2
    try:
        del fields['id']
    except KeyError:
        pass
    return Document(id=id, **fields)