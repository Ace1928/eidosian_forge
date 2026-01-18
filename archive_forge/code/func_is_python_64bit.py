import sys
import struct
import os
import threading
def is_python_64bit():
    return struct.calcsize('P') == 8