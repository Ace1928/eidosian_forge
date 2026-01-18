from __future__ import annotations
from ..runtime import driver
def get_ids_of_tensormaps(tensormaps_info):
    ret = None
    if tensormaps_info is not None:
        ret = [e.get_id_of_tensormap() for e in tensormaps_info]
    return ret