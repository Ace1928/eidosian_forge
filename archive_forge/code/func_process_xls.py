import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
def process_xls(self, link: str) -> str:
    import io
    import os
    try:
        import xlrd
    except ImportError:
        raise ImportError('`xlrd` package not found, please run `pip install xlrd`')
    try:
        import pandas as pd
    except ImportError:
        raise ImportError('`pandas` package not found, please run `pip install pandas`')
    response = self.confluence.request(path=link, absolute=True)
    text = ''
    if response.status_code != 200 or response.content == b'' or response.content is None:
        return text
    filename = os.path.basename(link)
    file_extension = os.path.splitext(filename)[1]
    if file_extension.startswith('.csv'):
        content_string = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(content_string))
        text += df.to_string(index=False, header=False) + '\n\n'
    else:
        workbook = xlrd.open_workbook(file_contents=response.content)
        for sheet in workbook.sheets():
            text += f'{sheet.name}:\n'
            for row in range(sheet.nrows):
                for col in range(sheet.ncols):
                    text += f'{sheet.cell_value(row, col)}\t'
                text += '\n'
            text += '\n'
    return text