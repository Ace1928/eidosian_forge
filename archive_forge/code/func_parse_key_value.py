import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
def parse_key_value(self):
    """Parse key-value pair separated by colon."""
    key, value = self.line.split(':', 1)
    return (key.strip(), value.strip())