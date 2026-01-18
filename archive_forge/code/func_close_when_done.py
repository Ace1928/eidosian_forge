import asyncore
from collections import deque
from warnings import _deprecated
def close_when_done(self):
    """automatically close this channel once the outgoing queue is empty"""
    self.producer_fifo.append(None)