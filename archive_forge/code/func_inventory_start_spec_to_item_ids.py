import collections
from typing import Tuple, Optional, Sequence, Dict, Set, List
def inventory_start_spec_to_item_ids(inv_spec: Sequence[dict]) -> List[str]:
    """Converts the argument of SimpleInventoryAgentStart into a list of equivalent
    item ids suitable for passing into other handlers, like FlatInventoryObservation and
    EquipAction.
    [dict(type=planks, metadata=2, quantity=3),
     dict(type=wooden_pickaxe, quantity=1), ...] => ["planks#2", "wooden_pickaxe", ...]
    """
    result = []
    for d in inv_spec:
        item_type = d['type']
        metadata = d.get('metadata')
        item_id = encode_item_with_metadata(item_type, metadata)
        result.append(item_id)
    return list(set(result))