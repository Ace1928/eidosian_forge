import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
@property
def bugs_closed(self):
    """ List of (Debian) bugs closed by the block """
    return self._get_bugs_closed_generic(closes)