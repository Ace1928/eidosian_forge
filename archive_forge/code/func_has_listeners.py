import asyncio
import logging
import threading
import weakref
from asgiref.sync import async_to_sync, iscoroutinefunction, sync_to_async
from django.utils.inspect import func_accepts_kwargs
def has_listeners(self, sender=None):
    sync_receivers, async_receivers = self._live_receivers(sender)
    return bool(sync_receivers) or bool(async_receivers)