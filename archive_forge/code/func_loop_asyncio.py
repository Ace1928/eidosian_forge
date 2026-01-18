import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@register_integration('asyncio')
def loop_asyncio(kernel):
    """Start a kernel with asyncio event loop support."""
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        return
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop._should_close = False

    def process_stream_events(stream):
        """fall back to main loop when there's a socket event"""
        if stream.flush(limit=1):
            loop.stop()
    notifier = partial(process_stream_events, kernel.shell_stream)
    loop.add_reader(kernel.shell_stream.getsockopt(zmq.FD), notifier)
    loop.call_soon(notifier)
    while True:
        error = None
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            continue
        except Exception as e:
            error = e
        if loop._should_close:
            loop.close()
        if error is not None:
            raise error
        break