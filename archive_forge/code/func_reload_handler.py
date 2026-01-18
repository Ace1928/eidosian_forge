import os
import sys
import time
import errno
import socket
import signal
import logging
import threading
import traceback
import email.message
import pyzor.config
import pyzor.account
import pyzor.engines.common
import pyzor.hacks.py26
def reload_handler(self, *args, **kwargs):
    """Handler for the SIGUSR1 signal. This should be used to reload
        the configuration files.
        """
    self.log.info('SIGUSR1 received. Reloading configuration.')
    t = threading.Thread(target=self.load_config)
    t.start()