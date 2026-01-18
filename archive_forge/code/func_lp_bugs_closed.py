import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
@property
def lp_bugs_closed(self):
    """ List of Launchpad bugs closed by the block """
    return self._get_bugs_closed_generic(closeslp)