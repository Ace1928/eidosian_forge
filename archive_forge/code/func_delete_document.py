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
def delete_document(self, doc_id, conn=None, delete_actual_document=False):
    """
        Delete a document from index
        Returns 1 if the document was deleted, 0 if not

        ### Parameters

        - **delete_actual_document**: if set to True, RediSearch also delete
                                      the actual document if it is in the index
        """
    args = [DEL_CMD, self.index_name, doc_id]
    if delete_actual_document:
        args.append('DD')
    if conn is not None:
        return conn.execute_command(*args)
    return self.execute_command(*args)