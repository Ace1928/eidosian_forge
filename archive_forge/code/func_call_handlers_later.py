from typing import List
from jupyter_client.channelsabc import HBChannelABC
def call_handlers_later(self, *args, **kwds):
    """Call the message handlers later.

        The default implementation just calls the handlers immediately, but this
        method exists so that GUI toolkits can defer calling the handlers until
        after the event loop has run, as expected by GUI frontends.
        """
    self.call_handlers(*args, **kwds)