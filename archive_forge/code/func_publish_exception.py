from typing import AsyncIterator, Dict, Optional, Union
import asyncio
import duet
import google.api_core.exceptions as google_exceptions
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
def publish_exception(self, exception: BaseException) -> None:
    """Publishes an exception to all outstanding futures."""
    for future in self._subscribers.values():
        if not future.done():
            future.set_exception(exception)
    self._subscribers.clear()