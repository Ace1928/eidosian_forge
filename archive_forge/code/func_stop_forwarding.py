import logging
import threading
def stop_forwarding(self):
    """disable forwarding and tell the forwarding thread to end itself"""
    self.forwarding_client = None