import asyncore
from collections import deque
from warnings import _deprecated
def push_with_producer(self, producer):
    self.producer_fifo.append(producer)
    self.initiate_send()