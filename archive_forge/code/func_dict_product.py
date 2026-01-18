import itertools
from typing import Any, Dict, Iterable, List, Tuple
def dict_product(d: Dict[str, Iterable[Any]], safe: bool=True) -> Iterable[Dict[str, Any]]:
    keys = d.keys()
    arrays = list(d.values())
    if len(arrays) == 0:
        if safe:
            yield {}
        return
    for element in _safe_product(arrays, safe):
        yield {k: v for k, v in zip(keys, element) if v is not _EMPTY_ITER}