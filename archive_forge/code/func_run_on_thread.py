import asyncio
import threading
from typing import List, Union, Any, TypeVar, Generic, Optional, Callable, Awaitable
from unittest.mock import AsyncMock
def run_on_thread(func: Callable[[], T]) -> T:
    box = Box()

    def set_box():
        box.val = func()
    thread = threading.Thread(target=set_box)
    thread.start()
    thread.join()
    return box.val