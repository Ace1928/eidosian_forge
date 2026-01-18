from typing import Any, Dict, Optional, Union, Iterable, List, Type, TYPE_CHECKING
from lazyops.utils.lazy import get_keydb_session
from .base import BaseStatefulBackend, SchemaType, logger
def get_all_values(self, **kwargs) -> List[Any]:
    """
        Returns all the Values
        """
    if not self.base_key:
        raise NotImplementedError('Cannot get all values from a Redis Cache without a base key')
    if self.hset_enabled:
        data = self.cache.hgetall(self.base_key)
        results = []
        for key, value in data.items():
            try:
                results.append(self.decode_value(value, **kwargs))
            except Exception as e:
                logger.warning(f'Unable to decode value for {key}: {e}')
                self.delete(key)
        return results
    keys = self._fetch_set_keys(decode=False)
    data_list = self.cache.client.mget(keys)
    results = []
    for key, value in zip(keys, data_list):
        try:
            results.append(self.decode_value(value, **kwargs))
        except Exception as e:
            logger.warning(f'Unable to decode value for {key}: {e}')
            self.delete(key)
    return results