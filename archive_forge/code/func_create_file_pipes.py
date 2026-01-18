import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def create_file_pipes(self):
    r, w = os.pipe()
    rf = os.fdopen(r, 'rb')
    wf = os.fdopen(w, 'wb')
    return (rf, wf)