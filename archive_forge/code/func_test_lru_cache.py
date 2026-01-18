from collections import OrderedDict
from torch import nn
from vllm.utils import LRUCache
from vllm.lora.utils import (parse_fine_tuned_lora_name, replace_submodule)
def test_lru_cache():
    cache = TestLRUCache(3)
    cache.put(1, 1)
    assert len(cache) == 1
    cache.put(1, 1)
    assert len(cache) == 1
    cache.put(2, 2)
    assert len(cache) == 2
    cache.put(3, 3)
    assert len(cache) == 3
    assert set(cache.cache) == {1, 2, 3}
    cache.put(4, 4)
    assert len(cache) == 3
    assert set(cache.cache) == {2, 3, 4}
    assert cache._remove_counter == 1
    assert cache.get(2) == 2
    cache.put(5, 5)
    assert set(cache.cache) == {2, 4, 5}
    assert cache._remove_counter == 2
    assert cache.pop(5) == 5
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache.pop(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache.get(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache.put(6, 6)
    assert len(cache) == 3
    assert set(cache.cache) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache
    cache.remove_oldest()
    assert len(cache) == 2
    assert set(cache.cache) == {2, 6}
    assert cache._remove_counter == 4
    cache.clear()
    assert len(cache) == 0
    assert cache._remove_counter == 6
    cache._remove_counter = 0
    cache[1] = 1
    assert len(cache) == 1
    cache[1] = 1
    assert len(cache) == 1
    cache[2] = 2
    assert len(cache) == 2
    cache[3] = 3
    assert len(cache) == 3
    assert set(cache.cache) == {1, 2, 3}
    cache[4] = 4
    assert len(cache) == 3
    assert set(cache.cache) == {2, 3, 4}
    assert cache._remove_counter == 1
    assert cache[2] == 2
    cache[5] = 5
    assert set(cache.cache) == {2, 4, 5}
    assert cache._remove_counter == 2
    del cache[5]
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache.pop(10)
    assert len(cache) == 2
    assert set(cache.cache) == {2, 4}
    assert cache._remove_counter == 3
    cache[6] = 6
    assert len(cache) == 3
    assert set(cache.cache) == {2, 4, 6}
    assert 2 in cache
    assert 4 in cache
    assert 6 in cache