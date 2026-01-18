from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import srsly
from preshed.bloom import BloomFilter
from .errors import Errors
from .strings import get_string_id
from .util import SimpleFrozenDict, ensure_path, load_language_data, registry
def remove_table(self, name: str) -> Table:
    """Remove a table. Raises an error if the table doesn't exist.

        name (str): Name of the table to remove.
        RETURNS (Table): The removed table.

        DOCS: https://spacy.io/api/lookups#remove_table
        """
    if name not in self._tables:
        raise KeyError(Errors.E159.format(name=name, tables=self.tables))
    return self._tables.pop(name)