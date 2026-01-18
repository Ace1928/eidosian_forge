import asyncio
import logging
import threading
import weakref
from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async
from django.utils.inspect import func_accepts_kwargs
@sync_to_async
def sync_send():
    responses = []
    for receiver in sync_receivers:
        try:
            response = receiver(signal=self, sender=sender, **named)
        except Exception as err:
            self._log_robust_failure(receiver, err)
            responses.append((receiver, err))
        else:
            responses.append((receiver, response))
    return responses