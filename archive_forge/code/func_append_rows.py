from __future__ import annotations
import os
import abc
from lazyops.libs.pooler import ThreadPooler
from typing import Optional, List, Dict, Any, Union, Type, Tuple, Generator, TYPE_CHECKING
from lazyops.imports._gspread import resolve_gspread
import gspread
def append_rows(self, data: List[List[Any]], sheet_name: Optional[str]=None) -> None:
    """
        Append rows into the worksheet
        """
    if not isinstance(data, list) or not isinstance(data[0], (list, dict)):
        return self.append_row(data, sheet_name=sheet_name)
    data = [list(d.values()) if isinstance(d, dict) else d for d in data]
    wks = self.get_worksheet(sheet_name)
    wks.append_rows(data)