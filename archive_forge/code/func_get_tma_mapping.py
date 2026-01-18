from __future__ import annotations
from ..runtime import driver
def get_tma_mapping(tensormaps_info):
    ret = {}
    if tensormaps_info is not None:
        for i, e in enumerate(tensormaps_info):
            ret.update(e.get_address_tma_mapping())
    else:
        ret = None
    return ret