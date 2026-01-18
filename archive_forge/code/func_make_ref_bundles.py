from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List
import ray
from ray.data.block import BlockAccessor, CallableClass
def make_ref_bundles(simple_data: List[List[Any]]) -> List['RefBundle']:
    """Create ref bundles from a list of block data.

    One bundle is created for each input block.
    """
    import pandas as pd
    from ray.data._internal.execution.interfaces import RefBundle
    output = []
    for block in simple_data:
        block = pd.DataFrame({'id': block})
        output.append(RefBundle([(ray.put(block), BlockAccessor.for_block(block).get_metadata([], None))], owns_blocks=True))
    return output