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
def create_receipt(self, user_id, methods, expires_at):
    """Given a set of payload attributes, generate a Fernet receipt."""
    payload = ReceiptPayload.assemble(user_id, methods, expires_at)
    serialized_payload = msgpack.packb(payload)
    receipt = self.pack(serialized_payload)
    if len(receipt) > 255:
        LOG.info('Fernet receipt created with length of %d characters, which exceeds 255 characters', len(receipt))
    return receipt