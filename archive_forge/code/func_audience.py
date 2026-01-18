import datetime
import io
import json
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
from google.oauth2 import utils
@property
def audience(self):
    """Optional[str]: The STS audience which contains the resource name for the
            workforce pool and the provider identifier in that pool."""
    return self._audience