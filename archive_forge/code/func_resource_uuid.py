import base64
import collections.abc
import contextlib
import grp
import hashlib
import itertools
import os
import pwd
import uuid
from cryptography import x509
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import urllib
from keystone.common import password_hashing
import keystone.conf
from keystone import exception
from keystone.i18n import _
def resource_uuid(value):
    """Convert input to valid UUID hex digits."""
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        if len(value) <= 64:
            return uuid.uuid5(RESOURCE_ID_NAMESPACE, value).hex
        raise ValueError(_('Length of transformable resource id > 64, which is max allowed characters'))