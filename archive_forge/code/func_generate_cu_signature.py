from __future__ import annotations
from ..runtime import driver
def generate_cu_signature(constants, signature, ids):
    num_regular_signatures = max(signature.keys()) + 1 if len(signature) > 0 else 0
    if ids['ids_of_tensormaps'] is not None:
        for i, _ in enumerate(ids['ids_of_tensormaps']):
            signature[num_regular_signatures + i] = '*CUtensorMap'
    return (signature, num_regular_signatures)