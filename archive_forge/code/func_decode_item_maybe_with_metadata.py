import collections
from typing import Tuple, Optional, Sequence, Dict, Set, List
def decode_item_maybe_with_metadata(s: str) -> Tuple[str, Optional[int]]:
    assert len(s) > 0
    if '#' in s:
        item_type, metadata_str = s.split('#')
        if len(item_type) == 0:
            raise ValueError(f"invalid item_type '{item_type}' in '{s}'")
        metadata = int(metadata_str)
        if metadata not in range(16):
            raise ValueError(f"invalid metadata {metadata} in '{s}'")
        return (item_type, int(metadata_str))
    return (s, None)