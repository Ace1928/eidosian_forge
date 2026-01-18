import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def set_subprocess(_subprocess=None):
    """Set subprocess module to use.

    The subprocess could be eventlet.green.subprocess if using eventlet,
    or Python's subprocess otherwise.
    """
    global subprocess
    subprocess = _subprocess