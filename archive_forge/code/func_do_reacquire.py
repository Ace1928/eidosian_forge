import threading
import time as mod_time
import uuid
from types import SimpleNamespace, TracebackType
from typing import Optional, Type
from redis.exceptions import LockError, LockNotOwnedError
from redis.typing import Number
def do_reacquire(self) -> bool:
    timeout = int(self.timeout * 1000)
    if not bool(self.lua_reacquire(keys=[self.name], args=[self.local.token, timeout], client=self.redis)):
        raise LockNotOwnedError("Cannot reacquire a lock that's no longer owned", lock_name=self.name)
    return True