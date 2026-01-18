import base64
import datetime
import struct
import uuid
from cryptography import fernet
import msgpack
from oslo_log import log
from oslo_utils import timeutils
from keystone.auth import plugins as auth_plugins
from keystone.common import fernet_utils as utils
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
@classmethod
def restore_padding(cls, receipt):
    """Restore padding based on receipt size.

        :param receipt: receipt to restore padding on
        :type receipt: str
        :returns: receipt with correct padding

        """
    mod_returned = len(receipt) % 4
    if mod_returned:
        missing_padding = 4 - mod_returned
        receipt += '=' * missing_padding
    return receipt