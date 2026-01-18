from __future__ import annotations
from contextlib import suppress
import sys
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs.indexing import NDFrameIndexerBase
from pandas._libs.lib import item_from_zerodim
from pandas.compat import PYPY
from pandas.errors import (
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import algorithms as algos
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.indexes.api import (
def convert_from_missing_indexer_tuple(indexer, axes):
    """
    Create a filtered indexer that doesn't have any missing indexers.
    """

    def get_indexer(_i, _idx):
        return axes[_i].get_loc(_idx['key']) if isinstance(_idx, dict) else _idx
    return tuple((get_indexer(_i, _idx) for _i, _idx in enumerate(indexer)))