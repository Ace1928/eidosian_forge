import logging
from typing import Optional, Tuple
import concurrent.futures
import ray.dashboard.modules.log.log_utils as log_utils
import ray.dashboard.modules.log.log_consts as log_consts
import ray.dashboard.utils as dashboard_utils
import ray.dashboard.optional_utils as dashboard_optional_utils
from ray._private.ray_constants import env_integer
import asyncio
import grpc
import io
import os
from pathlib import Path
from ray.core.generated import reporter_pb2
from ray.core.generated import reporter_pb2_grpc
from ray._private.ray_constants import (
def find_offset_of_content_in_file(file: io.BufferedIOBase, content: bytes, start_offset: int=0) -> int:
    """Find the offset of the first occurrence of content in a file.

    Args:
        file: File object
        content: Content to find
        start_offset: Start offset to read from, inclusive.

    Returns:
        Offset of the first occurrence of content in a file.
    """
    logger.debug(f'Finding offset of content {content} in file')
    file.seek(start_offset, io.SEEK_SET)
    offset = start_offset
    while True:
        block_data = file.read(BLOCK_SIZE)
        if block_data == b'':
            return -1
        block_offset = block_data.find(content)
        if block_offset != -1:
            return offset + block_offset
        offset += len(block_data)