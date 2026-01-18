import logging
import sys
import threading
from oslo_rootwrap import cmd
from oslo_rootwrap import subprocess
def forwarding_popen(f, old_popen=subprocess.Popen):

    def popen(*args, **kwargs):
        p = old_popen(*args, **kwargs)
        t = threading.Thread(target=forward_stream, args=(p.stderr, f))
        t.daemon = True
        t.start()
        return p
    return popen