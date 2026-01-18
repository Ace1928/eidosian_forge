from __future__ import annotations
import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union
import aiohttp
import requests
from aiohttp import ServerTimeoutError
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from requests.exceptions import Timeout
def json_to_md(json_contents: List[Dict[str, Union[str, int, float]]], table_name: Optional[str]=None) -> str:
    """Converts a JSON object to a markdown table."""
    if len(json_contents) == 0:
        return ''
    output_md = ''
    headers = json_contents[0].keys()
    for header in headers:
        header.replace('[', '.').replace(']', '')
        if table_name:
            header.replace(f'{table_name}.', '')
        output_md += f'| {header} '
    output_md += '|\n'
    for row in json_contents:
        for value in row.values():
            output_md += f'| {value} '
        output_md += '|\n'
    return output_md