import ctypes
import ctypes.util
import os
import sys
import threading
import time
def monotonic():
    """Monotonic clock, cannot go backward."""
    global get_tick_count_last_sample
    global get_tick_count_wraparounds
    with get_tick_count_lock:
        current_sample = GetTickCount()
        if current_sample < get_tick_count_last_sample:
            get_tick_count_wraparounds += 1
        get_tick_count_last_sample = current_sample
        final_milliseconds = get_tick_count_wraparounds << 32
        final_milliseconds += get_tick_count_last_sample
        return final_milliseconds / 1000.0