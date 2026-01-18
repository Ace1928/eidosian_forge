import os
import sys
import linecache
import re
import inspect
def hub_timer_stacks(state=False):
    """Toggles whether or not the hub records the stack when timers are set.
    To inspect the stacks of the current timers, call :func:`format_hub_timers`
    at critical junctures in the application logic.
    """
    from eventlet.hubs import timer
    timer._g_debug = state