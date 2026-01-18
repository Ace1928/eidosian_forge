from datetime import datetime
import io
import json
import logging
import six
from google.auth import _cloud_sdk
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import reauth
@classmethod
def from_authorized_user_file(cls, filename, scopes=None):
    """Creates a Credentials instance from an authorized user json file.

        Args:
            filename (str): The path to the authorized user json file.
            scopes (Sequence[str]): Optional list of scopes to include in the
                credentials.

        Returns:
            google.oauth2.credentials.Credentials: The constructed
                credentials.

        Raises:
            ValueError: If the file is not in the expected format.
        """
    with io.open(filename, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        return cls.from_authorized_user_info(data, scopes)