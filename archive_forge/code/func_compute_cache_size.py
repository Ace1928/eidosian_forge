import logging
import types
import weakref
from dataclasses import dataclass
from . import config
def compute_cache_size(frame: types.FrameType, cache_entry) -> CacheSizeRelevantForFrame:
    num_cache_entries = 0
    num_cache_entries_in_bucket = 0
    while cache_entry:
        num_cache_entries += 1
        if _is_same_cache_bucket(frame, cache_entry):
            num_cache_entries_in_bucket += 1
        cache_entry = cache_entry.next
    return CacheSizeRelevantForFrame(num_cache_entries, num_cache_entries_in_bucket)