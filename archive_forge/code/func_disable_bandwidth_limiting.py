import time
import threading
def disable_bandwidth_limiting(self):
    """Disable bandwidth limiting on reads to the stream"""
    self._bandwidth_limiting_enabled = False