from __future__ import annotations
import asyncio
import urllib.parse
from uvicorn._types import WWWScope
def get_path_with_query_string(scope: WWWScope) -> str:
    path_with_query_string = urllib.parse.quote(scope['path'])
    if scope['query_string']:
        path_with_query_string = '{}?{}'.format(path_with_query_string, scope['query_string'].decode('ascii'))
    return path_with_query_string