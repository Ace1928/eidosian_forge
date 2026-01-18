from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
def scan_for_short_id(object_store, prefix):
    """Scan an object store for a short id."""
    ret = []
    for object_id in object_store:
        if object_id.startswith(prefix):
            ret.append(object_store[object_id])
    if not ret:
        raise KeyError(prefix)
    if len(ret) == 1:
        return ret[0]
    raise AmbiguousShortId(prefix, ret)