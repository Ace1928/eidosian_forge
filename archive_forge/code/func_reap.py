import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
def reap(self):
    self[:] = (thread for thread in self if thread.is_alive())