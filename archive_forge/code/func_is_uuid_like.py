import hashlib
import logging
import uuid
from oslo_concurrency import lockutils
from oslo_utils.secretutils import md5
from glance_store.i18n import _
def is_uuid_like(val):
    """Returns validation of a value as a UUID.

    For our purposes, a UUID is a canonical form string:
    aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa
    """
    try:
        return str(uuid.UUID(val)) == val
    except (TypeError, ValueError, AttributeError):
        return False