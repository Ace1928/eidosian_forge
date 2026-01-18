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
def restore_spilled_objects(self, object_refs: List[ObjectRef], url_with_offset_list: List[str]):
    from smart_open import open
    total = 0
    for i in range(len(object_refs)):
        object_ref = object_refs[i]
        url_with_offset = url_with_offset_list[i].decode()
        parsed_result = parse_url_with_offset(url_with_offset)
        base_url = parsed_result.base_url
        offset = parsed_result.offset
        with open(base_url, 'rb', transport_params=self.transport_params) as f:
            f.seek(offset)
            address_len = int.from_bytes(f.read(8), byteorder='little')
            metadata_len = int.from_bytes(f.read(8), byteorder='little')
            buf_len = int.from_bytes(f.read(8), byteorder='little')
            self._size_check(address_len, metadata_len, buf_len, parsed_result.size)
            owner_address = f.read(address_len)
            total += buf_len
            metadata = f.read(metadata_len)
            self._put_object_to_store(metadata, buf_len, f, object_ref, owner_address)
    return total