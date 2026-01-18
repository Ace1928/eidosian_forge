import os
def set_ctime_windows(filepath, timestamp):
    if not win32_setctime.SUPPORTED:
        return
    try:
        win32_setctime.setctime(filepath, timestamp)
    except (OSError, ValueError):
        pass