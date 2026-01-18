import datetime
import sys
import threading
import time
import cherrypy
from cherrypy.lib import cptools, httputil
def expire_cache(self):
    """Continuously examine cached objects, expiring stale ones.

        This function is designed to be run in its own daemon thread,
        referenced at ``self.expiration_thread``.
        """
    while time:
        now = time.time()
        for expiration_time, objects in self.expirations.copy().items():
            if expiration_time <= now:
                for obj_size, uri, sel_header_values in objects:
                    try:
                        del self.store[uri][tuple(sel_header_values)]
                        self.tot_expires += 1
                        self.cursize -= obj_size
                    except KeyError:
                        pass
                del self.expirations[expiration_time]
        time.sleep(self.expire_freq)