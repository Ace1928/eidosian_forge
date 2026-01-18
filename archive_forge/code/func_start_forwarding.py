import logging
import threading
def start_forwarding(self):
    """start the forwarding thread"""
    threading.Thread(target=self._forward_loop).start()