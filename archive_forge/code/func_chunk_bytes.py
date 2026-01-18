import asyncio
import logging
import sys
import tempfile
import traceback
from contextlib import aclosing
from asgiref.sync import ThreadSensitiveContext, sync_to_async
from django.conf import settings
from django.core import signals
from django.core.exceptions import RequestAborted, RequestDataTooBig
from django.core.handlers import base
from django.http import (
from django.urls import set_script_prefix
from django.utils.functional import cached_property
@classmethod
def chunk_bytes(cls, data):
    """
        Chunks some data up so it can be sent in reasonable size messages.
        Yields (chunk, last_chunk) tuples.
        """
    position = 0
    if not data:
        yield (data, True)
        return
    while position < len(data):
        yield (data[position:position + cls.chunk_size], position + cls.chunk_size >= len(data))
        position += cls.chunk_size