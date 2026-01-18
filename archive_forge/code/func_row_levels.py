from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@property
def row_levels(self) -> int:
    if self.fmt.index:
        return self.frame.index.nlevels
    elif self.show_col_idx_names:
        return 1
    return 0