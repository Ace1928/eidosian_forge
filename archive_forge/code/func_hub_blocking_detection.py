import os
import sys
import linecache
import re
import inspect
def hub_blocking_detection(state=False, resolution=1):
    """Toggles whether Eventlet makes an effort to detect blocking
    behavior in an application.

    It does this by telling the kernel to raise a SIGALARM after a
    short timeout, and clearing the timeout every time the hub
    greenlet is resumed.  Therefore, any code that runs for a long
    time without yielding to the hub will get interrupted by the
    blocking detector (don't use it in production!).

    The *resolution* argument governs how long the SIGALARM timeout
    waits in seconds.  The implementation uses :func:`signal.setitimer`
    and can be specified as a floating-point value.
    The shorter the resolution, the greater the chance of false
    positives.
    """
    from eventlet import hubs
    assert resolution > 0
    hubs.get_hub().debug_blocking = state
    hubs.get_hub().debug_blocking_resolution = resolution
    if not state:
        hubs.get_hub().block_detect_post()