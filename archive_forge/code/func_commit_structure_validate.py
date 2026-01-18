import asyncio
import os
import weakref
from asyncio import AbstractEventLoop
from types import MethodType
from typing import Any, Awaitable, Coroutine, Dict, Tuple, TypeVar, Union, cast
import async_timeout
from packaging.version import Version
from .structs import OffsetAndMetadata, TopicPartition
def commit_structure_validate(offsets: Dict[TopicPartition, Union[int, Tuple[int, str], OffsetAndMetadata]]) -> Dict[TopicPartition, OffsetAndMetadata]:
    if not offsets or not isinstance(offsets, dict):
        raise ValueError(offsets)
    formatted_offsets = {}
    for tp, offset_and_metadata in offsets.items():
        if not isinstance(tp, TopicPartition):
            raise ValueError('Key should be TopicPartition instance')
        if isinstance(offset_and_metadata, int):
            offset, metadata = (offset_and_metadata, '')
        else:
            try:
                offset, metadata = offset_and_metadata
            except Exception:
                raise ValueError(offsets)
            if not isinstance(metadata, str):
                raise ValueError('Metadata should be a string')
        formatted_offsets[tp] = OffsetAndMetadata(offset, metadata)
    return formatted_offsets