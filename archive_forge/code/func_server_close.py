import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
def server_close(self):
    super().server_close()
    self.collect_children(blocking=self.block_on_close)