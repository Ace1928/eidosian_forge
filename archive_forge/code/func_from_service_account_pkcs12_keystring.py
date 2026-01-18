from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from google.auth import _helpers
from google.auth.crypt import base as crypt_base
from google.oauth2 import service_account
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
@classmethod
def from_service_account_pkcs12_keystring(cls, key_string, password=None, **kwargs):
    password = password or _DEFAULT_PASSWORD
    signer = PKCS12Signer.from_string((key_string, password))
    missing_fields = [f for f in cls._REQUIRED_FIELDS if f not in kwargs]
    if missing_fields:
        raise MissingRequiredFieldsError('Missing fields: {}.'.format(', '.join(missing_fields)))
    creds = cls(signer, **kwargs)
    creds._private_key_pkcs12 = key_string
    creds._private_key_password = password
    return creds