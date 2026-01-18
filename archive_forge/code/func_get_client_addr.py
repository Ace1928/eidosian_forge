from __future__ import annotations
import asyncio
import urllib.parse
from uvicorn._types import WWWScope
def get_client_addr(scope: WWWScope) -> str:
    client = scope.get('client')
    if not client:
        return ''
    return '%s:%d' % client