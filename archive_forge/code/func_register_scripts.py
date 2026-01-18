import asyncio
import threading
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, NoReturn, Optional, Union
from aioredis.exceptions import LockError, LockNotOwnedError
def register_scripts(self):
    cls = self.__class__
    client = self.redis
    if cls.lua_release is None:
        cls.lua_release = client.register_script(cls.LUA_RELEASE_SCRIPT)
    if cls.lua_extend is None:
        cls.lua_extend = client.register_script(cls.LUA_EXTEND_SCRIPT)
    if cls.lua_reacquire is None:
        cls.lua_reacquire = client.register_script(cls.LUA_REACQUIRE_SCRIPT)