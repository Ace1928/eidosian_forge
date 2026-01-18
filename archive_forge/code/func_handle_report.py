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
def handle_report(self, digests):
    """Handle the 'report' command in a single step.

        This command increases the spam count for the specified digests."""
    self.server.log.debug('Request to report digests %s', digests)
    if self.server.one_step:
        self.server.database.report(digests)
    else:
        for digest in digests:
            try:
                record = self.server.database[digest]
            except KeyError:
                record = pyzor.engines.common.Record()
            record.r_increment()
            self.server.database[digest] = record
    if self.server.forwarder:
        for digest in digests:
            self.server.forwarder.queue_forward_request(digest)