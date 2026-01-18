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
def handle_check(self, digests):
    """Handle the 'check' command.

        This command returns the spam/ham counts for the specified digest.
        """
    digest = digests[0]
    try:
        record = self.server.database[digest]
    except KeyError:
        record = pyzor.engines.common.Record()
    self.server.log.debug('Request to check digest %s', digest)
    self.response['Count'] = '%d' % record.r_count
    self.response['WL-Count'] = '%d' % record.wl_count