import re
from typing import Any, Dict, List, Optional, Tuple
from ase.db.core import Database, default_key_descriptions
from ase.db.table import Table, all_columns
@property
def row2(self) -> int:
    assert self.nrows is not None
    return min((self.page + 1) * self.limit, self.nrows)