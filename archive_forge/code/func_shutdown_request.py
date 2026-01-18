import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time
def shutdown_request(self, request):
    self.close_request(request)