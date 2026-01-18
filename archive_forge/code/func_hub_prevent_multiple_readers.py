import os
import sys
import linecache
import re
import inspect
def hub_prevent_multiple_readers(state=True):
    """Toggle prevention of multiple greenlets reading from a socket

    When multiple greenlets read from the same socket it is often hard
    to predict which greenlet will receive what data.  To achieve
    resource sharing consider using ``eventlet.pools.Pool`` instead.

    But if you really know what you are doing you can change the state
    to ``False`` to stop the hub from protecting against this mistake.
    """
    from eventlet.hubs import hub, get_hub
    from eventlet.hubs import asyncio
    if not state and isinstance(get_hub(), asyncio.Hub):
        raise RuntimeError('Multiple readers are not yet supported by asyncio hub')
    hub.g_prevent_multiple_readers = state