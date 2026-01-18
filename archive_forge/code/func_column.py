import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def column(self, *args, **kwargs):
    """
        Select a column by its column name, or numeric index.

        Args:
            i (`Union[int, str]`):
                The index or name of the column to retrieve.

        Returns:
            `pyarrow.ChunkedArray`
        """
    return self.table.column(*args, **kwargs)