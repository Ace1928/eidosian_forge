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
def handle_info(self, digests):
    """Handle the 'info' command.

        This command returns diagnostic data about a digest (timestamps for
        when the digest was first/last seen as spam/ham, and spam/ham
        counts).
        """
    digest = digests[0]
    try:
        record = self.server.database[digest]
    except KeyError:
        record = pyzor.engines.common.Record()
    self.server.log.debug('Request for information about digest %s', digest)

    def time_output(time_obj):
        """Convert a datetime object to a POSIX timestamp.

            If the object is None, then return 0.
            """
        if not time_obj:
            return 0
        return time.mktime(time_obj.timetuple())
    self.response['Entered'] = '%d' % time_output(record.r_entered)
    self.response['Updated'] = '%d' % time_output(record.r_updated)
    self.response['WL-Entered'] = '%d' % time_output(record.wl_entered)
    self.response['WL-Updated'] = '%d' % time_output(record.wl_updated)
    self.response['Count'] = '%d' % record.r_count
    self.response['WL-Count'] = '%d' % record.wl_count