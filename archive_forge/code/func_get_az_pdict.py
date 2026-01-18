from __future__ import annotations
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from lazyops.utils.logs import logger
from lazyops.utils.lazy import lazy_import
from lazyops.utils.helpers import fail_after
from typing import Any, Callable, Dict, List, Optional, Union, Type
def get_az_pdict(base_key: str, expiration: Optional[int]=None, aliases: Optional[List[str]]=None, hset_disabled: Optional[bool]=False, **kwargs) -> 'PersistentDict':
    """
    Lazily initializes the persistent dict
    """
    global _az_pdicts, _az_pdict_aliases
    if base_key not in _az_pdicts and base_key not in _az_pdict_aliases:
        s = get_az_settings()
        sess = get_az_kdb('persistence', serializer=None)
        if s.local_persistence_fallback:
            try:
                with fail_after(5):
                    sess.ping()
            except Exception as e:
                logger.warning(f'Failed to connect to KVDB persistence backend, falling back to local: {e}')
        if sess is not None:
            _az_pdicts[base_key] = sess.create_persistence(base_key=base_key, expiration=expiration, hset_disabled=hset_disabled, **kwargs)
        else:
            from lazyops.libs.persistence import PersistentDict
            _az_pdicts[base_key] = PersistentDict(base_key=base_key, expiration=expiration, hset_disabled=hset_disabled, file_path=s.data_dir.joinpath(f'{s.app_name}.cache'), **kwargs)
        if aliases:
            for alias in aliases:
                _az_pdict_aliases[alias] = base_key
    elif base_key in _az_pdict_aliases:
        base_key = _az_pdict_aliases[base_key]
    return _az_pdicts[base_key]