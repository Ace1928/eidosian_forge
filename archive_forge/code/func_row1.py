import re
from typing import Any, Dict, List, Optional, Tuple
from ase.db.core import Database, default_key_descriptions
from ase.db.table import Table, all_columns
@property
def row1(self) -> int:
    return self.page * self.limit + 1