import re
import itertools
import textwrap
import uuid
import param
from param.display import register_display_accessor
from param._utils import async_executor
def ipython_async_executor(func):
    event_loop = None
    try:
        ip = get_ipython()
        if ip.kernel:
            from tornado.ioloop import IOLoop
            ioloop = IOLoop.current()
            event_loop = ioloop.asyncio_loop
            if event_loop.is_running():
                ioloop.add_callback(func)
            else:
                event_loop.run_until_complete(func())
            return
    except (NameError, AttributeError):
        pass
    async_executor(func)