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
def convert_uuid_hex_to_bytes(cls, uuid_string):
    """Compress UUID formatted strings to bytes.

        :param uuid_string: uuid string to compress to bytes
        :returns: a byte representation of the uuid

        """
    uuid_obj = uuid.UUID(uuid_string)
    return uuid_obj.bytes