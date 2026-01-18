from __future__ import unicode_literals
import struct
from six import int2byte, binary_type, iterbytes
from .log import logger
def do_received(self, data):
    """ Received telnet DO command. """
    logger.info('DO %r', data)