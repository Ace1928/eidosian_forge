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
def convert_uuid_bytes_to_hex(cls, uuid_byte_string):
    """Generate uuid.hex format based on byte string.

        :param uuid_byte_string: uuid string to generate from
        :returns: uuid hex formatted string

        """
    uuid_obj = uuid.UUID(bytes=uuid_byte_string)
    return uuid_obj.hex