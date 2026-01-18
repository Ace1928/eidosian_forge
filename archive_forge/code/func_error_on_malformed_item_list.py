import collections
from typing import Tuple, Optional, Sequence, Dict, Set, List
def error_on_malformed_item_list(item_list: Sequence[str], special_items: Sequence[str]):
    """Check that there are no duplicate item IDs, overlapping item IDs
    (e.g. "planks#2" and "planks"), that special item types don't come with metadata
    constraints (e.g. "air#0" should not be valid).
    """
    map_type_to_metadata: Dict[str, Set[Optional[int]]] = collections.defaultdict(set)
    for s in item_list:
        item_type, metadata = decode_item_maybe_with_metadata(s)
        if item_type in special_items and metadata is not None:
            raise ValueError(f"Non-None metadata={metadata} is not allowed for special item type '{item_type}'")
        metadata_set = map_type_to_metadata[item_type]
        if metadata in metadata_set:
            raise ValueError(f"Duplicate item entry for item '{s}'")
        map_type_to_metadata[item_type].add(metadata)
    for item_type, metadata_set in map_type_to_metadata.items():
        if None in metadata_set and len(metadata_set) != 1:
            raise ValueError(f'Item list overlaps for item_type={item_type}. This item list includes both the wildcard metadata option and at least one specific metadata: {[n for n in metadata_set if n is not None]}')