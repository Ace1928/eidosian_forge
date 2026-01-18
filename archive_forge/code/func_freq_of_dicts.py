import logging
import time
from collections import Counter
from functools import reduce
from typing import Dict, List
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import (
from ray.core.generated.common_pb2 import PlacementStrategy
def freq_of_dicts(dicts: List[Dict], serializer=lambda d: frozenset(d.items()), deserializer=dict) -> DictCount:
    """Count a list of dictionaries (or unhashable types).

    This is somewhat annoying because mutable data structures aren't hashable,
    and set/dict keys must be hashable.

    Args:
        dicts (List[D]): A list of dictionaries to be counted.
        serializer (D -> S): A custom serialization function. The output type S
            must be hashable. The default serializer converts a dictionary into
            a frozenset of KV pairs.
        deserializer (S -> U): A custom deserialization function. See the
            serializer for information about type S. For dictionaries U := D.

    Returns:
        List[Tuple[U, int]]: Returns a list of tuples. Each entry in the list
            is a tuple containing a unique entry from `dicts` and its
            corresponding frequency count.
    """
    freqs = Counter((serializer(d) for d in dicts))
    as_list = []
    for as_set, count in freqs.items():
        as_list.append((deserializer(as_set), count))
    return as_list