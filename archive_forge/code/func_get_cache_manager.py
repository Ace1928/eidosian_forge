import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def get_cache_manager(key) -> CacheManager:
    import os
    user_cache_manager = os.environ.get('TRITON_CACHE_MANAGER', None)
    global __cache_cls
    global __cache_cls_nme
    if user_cache_manager is not None and user_cache_manager != __cache_cls_nme:
        import importlib
        module_path, clz_nme = user_cache_manager.split(':')
        module = importlib.import_module(module_path)
        __cache_cls = getattr(module, clz_nme)
        __cache_cls_nme = user_cache_manager
    return __cache_cls(key)