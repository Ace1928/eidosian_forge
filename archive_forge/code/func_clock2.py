import time
def clock2():
    """Under windows, system CPU time can't be measured.

        This just returns process_time() and zero."""
    return (time.process_time(), 0.0)