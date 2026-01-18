import requests
import aiohttp
import asyncio
from dataclasses import dataclass
from lazyops.lazyclasses import lazyclass
from typing import List, Dict, Any, Optional
@property
def sess(self):
    if not self.session:
        self.session = LazySession(header=self.header.config)
    return self.session