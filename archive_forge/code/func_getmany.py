import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
def getmany(self, names: List, return_first: bool=True):
    results = {}
    for name in names:
        if name in self:
            if return_first:
                return self[name]
            results[name] = self[name]
    return results