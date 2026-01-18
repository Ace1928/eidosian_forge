from __future__ import annotations
import os
import abc
from lazyops.libs.pooler import ThreadPooler
from typing import Optional, List, Dict, Any, Union, Type, Tuple, Generator, TYPE_CHECKING
from lazyops.imports._gspread import resolve_gspread
import gspread
def insert_rows(self, data: Union[List[Dict[str, Any]], List[List[Any]]], index: int=1, value_input_option: str='raw', inherit_from_before: bool=True, sheet_name: Optional[str]=None):
    """
        Insert rows into the worksheet
        """
    if not isinstance(data, list) or not isinstance(data[0], (list, dict)):
        return self.insert_row(data, index=index, value_input_option=value_input_option, inherit_from_before=inherit_from_before, sheet_name=sheet_name)
    data = [list(d.values()) if isinstance(d, dict) else d for d in data]
    wks = self.get_worksheet(sheet_name)
    wks.insert_rows(data, index=index, value_input_option=value_input_option, inherit_from_before=inherit_from_before)