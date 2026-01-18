from __future__ import annotations
import os
import abc
from lazyops.libs.pooler import ThreadPooler
from typing import Optional, List, Dict, Any, Union, Type, Tuple, Generator, TYPE_CHECKING
from lazyops.imports._gspread import resolve_gspread
import gspread
def get_worksheet(self, key: Optional[Union[str, int]]=None) -> 'gspread.worksheet.Worksheet':
    """
        Returns a sheet by name or index
        """
    if key is None:
        key = self.sheet_name
    if key == self.sheet_name:
        return self.wks
    if isinstance(key, int):
        return self.sheet.worksheet(self.worksheet_names[key])
    if isinstance(key, str) and key not in self.worksheet_names:
        return self.duplicate_worksheet(key)
    return self.sheet.worksheet(key)