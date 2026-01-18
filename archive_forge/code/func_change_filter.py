from __future__ import absolute_import
import codecs
import os
import sys
import threading
import serial
from serial.tools.list_ports import comports
from serial.tools import hexlify_codec
def change_filter(self):
    """change the i/o transformations"""
    sys.stderr.write('\n--- Available Filters:\n')
    sys.stderr.write('\n'.join(('---   {:<10} = {.__doc__}'.format(k, v) for k, v in sorted(TRANSFORMATIONS.items()))))
    sys.stderr.write('\n--- Enter new filter name(s) [{}]: '.format(' '.join(self.filters)))
    with self.console:
        new_filters = sys.stdin.readline().lower().split()
    if new_filters:
        for f in new_filters:
            if f not in TRANSFORMATIONS:
                sys.stderr.write('--- unknown filter: {!r}\n'.format(f))
                break
        else:
            self.filters = new_filters
            self.update_transformations()
    sys.stderr.write('--- filters: {}\n'.format(' '.join(self.filters)))