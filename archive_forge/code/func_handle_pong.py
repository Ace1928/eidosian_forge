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
def handle_pong(self, digests):
    """Handle the 'pong' command.

        This command returns maxint for report counts and 0 whitelist.
        """
    self.server.log.debug('Request pong for %s', digests[0])
    self.response['Count'] = '%d' % sys.maxint
    self.response['WL-Count'] = '%d' % 0