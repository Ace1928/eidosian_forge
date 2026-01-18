from __future__ import annotations
from ..runtime import driver
def dummy_tensormaps_info(n=2):
    ret = []
    for i in range(n):
        ret.append(InfoFromBackendForTensorMap(dummy=True))
    return ret