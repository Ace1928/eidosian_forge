import datetime
import os
import shutil
import tempfile
import time
import types
import warnings
from dulwich.tests import SkipTest
from ..index import commit_tree
from ..objects import Commit, FixedSha, Tag, object_class
from ..pack import (
from ..repo import Repo
def build_pack(f, objects_spec, store=None):
    """Write test pack data from a concise spec.

    Args:
      f: A file-like object to write the pack to.
      objects_spec: A list of (type_num, obj). For non-delta types, obj
        is the string of that object's data.
        For delta types, obj is a tuple of (base, data), where:

        * base can be either an index in objects_spec of the base for that
        * delta; or for a ref delta, a SHA, in which case the resulting pack
        * will be thin and the base will be an external ref.
        * data is a string of the full, non-deltified data for that object.

        Note that offsets/refs and deltas are computed within this function.
      store: An optional ObjectStore for looking up external refs.
    Returns: A list of tuples in the order specified by objects_spec:
        (offset, type num, data, sha, CRC32)
    """
    sf = SHA1Writer(f)
    num_objects = len(objects_spec)
    write_pack_header(sf.write, num_objects)
    full_objects = {}
    offsets = {}
    crc32s = {}
    while len(full_objects) < num_objects:
        for i, (type_num, data) in enumerate(objects_spec):
            if type_num not in DELTA_TYPES:
                full_objects[i] = (type_num, data, obj_sha(type_num, [data]))
                continue
            base, data = data
            if isinstance(base, int):
                if base not in full_objects:
                    continue
                base_type_num, _, _ = full_objects[base]
            else:
                base_type_num, _ = store.get_raw(base)
            full_objects[i] = (base_type_num, data, obj_sha(base_type_num, [data]))
    for i, (type_num, obj) in enumerate(objects_spec):
        offset = f.tell()
        if type_num == OFS_DELTA:
            base_index, data = obj
            base = offset - offsets[base_index]
            _, base_data, _ = full_objects[base_index]
            obj = (base, list(create_delta(base_data, data)))
        elif type_num == REF_DELTA:
            base_ref, data = obj
            if isinstance(base_ref, int):
                _, base_data, base = full_objects[base_ref]
            else:
                base_type_num, base_data = store.get_raw(base_ref)
                base = obj_sha(base_type_num, base_data)
            obj = (base, list(create_delta(base_data, data)))
        crc32 = write_pack_object(sf.write, type_num, obj)
        offsets[i] = offset
        crc32s[i] = crc32
    expected = []
    for i in range(num_objects):
        type_num, data, sha = full_objects[i]
        assert len(sha) == 20
        expected.append((offsets[i], type_num, data, sha, crc32s[i]))
    sf.write_sha()
    f.seek(0)
    return expected