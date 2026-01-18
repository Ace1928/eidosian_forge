import time
import dateparser
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Callable
def getorset(self, name, value=None):
    if name in self:
        return self[name]
    self[name] = value
    return value