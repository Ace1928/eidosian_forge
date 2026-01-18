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
@property
def private_key_password(self):
    return self._private_key_password