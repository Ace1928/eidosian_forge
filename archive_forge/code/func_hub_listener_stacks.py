import os
import sys
import linecache
import re
import inspect
def hub_listener_stacks(state=False):
    """Toggles whether or not the hub records the stack when clients register
    listeners on file descriptors.  This can be useful when trying to figure
    out what the hub is up to at any given moment.  To inspect the stacks
    of the current listeners, call :func:`format_hub_listeners` at critical
    junctures in the application logic.
    """
    from eventlet import hubs
    hubs.get_hub().set_debug_listeners(state)