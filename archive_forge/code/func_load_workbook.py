from __future__ import annotations
from datetime import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
import pandas as pd
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader
def load_workbook(self, filepath_or_buffer: FilePath | ReadBuffer[bytes], engine_kwargs: Any) -> CalamineWorkbook:
    from python_calamine import load_workbook
    return load_workbook(filepath_or_buffer, **engine_kwargs)