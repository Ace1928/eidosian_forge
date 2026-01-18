import sys
import struct
import os
import threading
def loop_in_thread():
    while True:
        import time
        time.sleep(0.5)
        sys.stdout.write('#')
        sys.stdout.flush()