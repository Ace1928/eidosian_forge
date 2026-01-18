import requests
import aiohttp
import asyncio
from dataclasses import dataclass
from lazyops.lazyclasses import lazyclass
from typing import List, Dict, Any, Optional
@property
def pkey(self):
    if self.method in ['POST']:
        return 'json'
    if self.method in ['GET']:
        return 'params'
    return 'data'