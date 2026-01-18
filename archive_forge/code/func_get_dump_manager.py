import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def get_dump_manager(key) -> CacheManager:
    return __cache_cls(key, dump=True)