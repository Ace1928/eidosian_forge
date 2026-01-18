import collections
from typing import Tuple, Optional, Sequence, Dict, Set, List
def get_unique_matching_item_list_id(item_list: Sequence[str], item_type: str, metadata: int, clobber_logs=True) -> Optional[str]:
    """
    Assuming that `item_list` doesn't have duplicate elements, returns the string
    identifier in `item_list` that corresponds to `item_type` and `metadata`.
    If there is no matching strings, returns None.
    If there are multiple matching strings, raises a ValueError, because item_lists should
    not have overlapping item identifiers.
    Args:
        item_list: A list of item identifiers. Either just the item type ("wooden_pickaxe")
            or the item type with a metadata requirement ("planks#2").
        item_type: The item type to search for.
        metadata: The metadata to search for.
        clobber_logs: If True, and the only ID in `item_list` starting with "log.*" is the
            ID "log" (no metadata suffix), then treat "log" as a match for "log2". This is
            useful for MineRLTreeChop-v0 where obtaining acacia logs ("log2") should also
            count as obtaining regular logs. Note that no clobbering will occur if any
            log item IDs include metadata, or there's any log2 item IDs in the `item_list`.
            A potentially less tricky solution than this clobbering
            behavior could be adding a special type "log*" type which matches both "log"
            and "log2".
    """
    assert metadata is not None
    if clobber_logs and item_type == 'log2':
        log_start = [x for x in item_list if x.startswith('log')]
        if len(log_start) == 1 and log_start[0] == 'log':
            item_type = 'log'
    result = None
    matches = 0
    if item_type in item_list:
        result = item_type
        matches += 1
    key = encode_item_with_metadata(item_type, metadata)
    if key in item_list:
        result = key
        matches += 1
    if matches > 1:
        raise ValueError(f'Multiple item identifiers match with {(item_type, metadata)}')
    return result