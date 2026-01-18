import collections
from typing import Tuple, Optional, Sequence, Dict, Set, List
def item_list_contains(item_list: Sequence[str], item_type: str, metadata: Optional[str]):
    if metadata is None:
        return item_type in item_list
    else:
        return encode_item_with_metadata(item_type, metadata) in item_list