from __future__ import annotations
import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Pattern
from urllib.parse import urlparse
import numpy as np
def check_redis_module_exist(client: RedisType, required_modules: List[dict]) -> None:
    """Check if the correct Redis modules are installed."""
    installed_modules = client.module_list()
    installed_modules = {module[b'name'].decode('utf-8'): module for module in installed_modules}
    for module in required_modules:
        if module['name'] in installed_modules and int(installed_modules[module['name']][b'ver']) >= int(module['ver']):
            return
    error_message = 'Redis cannot be used as a vector database without RediSearch >=2.4Please head to https://redis.io/docs/stack/search/quick_start/to know more about installing the RediSearch module within Redis Stack.'
    logger.error(error_message)
    raise ValueError(error_message)