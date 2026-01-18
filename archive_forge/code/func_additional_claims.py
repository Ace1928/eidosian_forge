import copy
import datetime
import json
import cachetools
import six
from six.moves import urllib
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials
@property
def additional_claims(self):
    """ Additional claims the JWT object was created with."""
    return self._additional_claims