import threading
import time
import lazy_loader as lazy
def import_np():
    time.sleep(0.5)
    lazy.load('numpy')