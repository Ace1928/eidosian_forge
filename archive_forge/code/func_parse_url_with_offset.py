import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
def parse_url_with_offset(url_with_offset: str) -> Tuple[str, int, int]:
    """Parse url_with_offset to retrieve information.

    base_url is the url where the object ref
    is stored in the external storage.

    Args:
        url_with_offset: url created by create_url_with_offset.

    Returns:
        named tuple of base_url, offset, and size.
    """
    parsed_result = urllib.parse.urlparse(url_with_offset)
    query_dict = urllib.parse.parse_qs(parsed_result.query)
    base_url = parsed_result.geturl().split('?')[0]
    if 'offset' not in query_dict or 'size' not in query_dict:
        raise ValueError(f'Failed to parse URL: {url_with_offset}')
    offset = int(query_dict['offset'][0])
    size = int(query_dict['size'][0])
    return ParsedURL(base_url=base_url, offset=offset, size=size)