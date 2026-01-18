from __future__ import annotations
import os
import abc
from lazyops.libs.pooler import ThreadPooler
from typing import Optional, List, Dict, Any, Union, Type, Tuple, Generator, TYPE_CHECKING
from lazyops.imports._gspread import resolve_gspread
import gspread
def insert_row(self, data: Union[Dict[str, Any], List[Any]], index: int=1, value_input_option: str='raw', inherit_from_before: bool=True, sheet_name: Optional[str]=None):
    """
        Insert a row into the worksheet
        """
    if isinstance(data, dict):
        data = list(data.values())
    wks = self.get_worksheet(sheet_name)
    wks.insert_row(data, index=index, value_input_option=value_input_option, inherit_from_before=inherit_from_before)