import json
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
def get_override_manager(key) -> CacheManager:
    return __cache_cls(key, override=True)