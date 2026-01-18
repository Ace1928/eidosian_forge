from __future__ import annotations
import os
import abc
from lazyops.libs.pooler import ThreadPooler
from typing import Optional, List, Dict, Any, Union, Type, Tuple, Generator, TYPE_CHECKING
from lazyops.imports._gspread import resolve_gspread
import gspread
def duplicate_worksheet(self, name: str, src_name: Optional[str]=None) -> 'gspread.worksheet.Worksheet':
    """
        Duplicate the current worksheet
        """
    src_name = src_name or self.sheet_name
    wks = self.sheet.worksheet(src_name)
    self._worksheet_names = None
    return wks.duplicate(new_sheet_name=name)