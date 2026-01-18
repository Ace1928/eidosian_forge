from typing import Dict, Tuple, Callable
def save_entry(module_name: str, name: str, cache: CacheValues) -> None:
    try:
        module_cache = _cache[module_name]
    except KeyError:
        module_cache = _cache[module_name] = {}
    module_cache[name] = cache