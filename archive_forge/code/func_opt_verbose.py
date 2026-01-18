import os
import traceback
from twisted.application import internet, service
from twisted.names import authority, dns, secondary, server
from twisted.python import usage
def opt_verbose(self):
    """Increment verbosity level"""
    self['verbose'] += 1