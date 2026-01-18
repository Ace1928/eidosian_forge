import threading
import time as mod_time
import uuid
from types import SimpleNamespace, TracebackType
from typing import Optional, Type
from redis.exceptions import LockError, LockNotOwnedError
from redis.typing import Number
def do_extend(self, additional_time: int, replace_ttl: bool) -> bool:
    additional_time = int(additional_time * 1000)
    if not bool(self.lua_extend(keys=[self.name], args=[self.local.token, additional_time, '1' if replace_ttl else '0'], client=self.redis)):
        raise LockNotOwnedError("Cannot extend a lock that's no longer owned", lock_name=self.name)
    return True