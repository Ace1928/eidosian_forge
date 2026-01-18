import requests
import aiohttp
import asyncio
from dataclasses import dataclass
from lazyops.lazyclasses import lazyclass
from typing import List, Dict, Any, Optional
def api_call(self, is_async=False, *args, **kwargs):
    return self.api(*args, route=self.config.default_async if is_async else self.config.default_fetch, **kwargs)