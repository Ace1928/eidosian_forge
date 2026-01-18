import logging
import os
from os_ken.lib import ip
def connect_loop(self, handle, interval):
    while self._is_active:
        sock = self.connect()
        if sock:
            handle(sock, self.addr)
        sleep(interval)