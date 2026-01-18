from __future__ import annotations
from abc import (
from collections import abc
from io import StringIO
from itertools import islice
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.json import (
from pandas._libs.tslibs import iNaT
from pandas.compat._optional import import_optional_dependency
from pandas.errors import AbstractMethodError
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.json._normalize import convert_to_line_delimits
from pandas.io.json._table_schema import (
from pandas.io.parsers.readers import validate_integer
@property
def obj_to_write(self) -> NDFrame | Mapping[IndexLabel, Any]:
    return {'schema': self.schema, 'data': self.obj}