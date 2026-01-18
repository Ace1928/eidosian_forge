import collections
from typing import Tuple, Optional, Sequence, Dict, Set, List
def encode_item_with_metadata(item_type: str, metadata: Optional[int]) -> str:
    assert len(item_type) > 0
    if metadata is None:
        return item_type
    else:
        assert isinstance(metadata, int)
        return f'{item_type}#{metadata}'