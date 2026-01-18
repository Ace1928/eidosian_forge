import abc
import io
import json
import six
from google.auth import exceptions
@classmethod
def from_service_account_info(cls, info):
    """Creates a Signer instance instance from a dictionary containing
        service account info in Google format.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.

        Returns:
            google.auth.crypt.Signer: The constructed signer.

        Raises:
            ValueError: If the info is not in the expected format.
        """
    if _JSON_FILE_PRIVATE_KEY not in info:
        raise exceptions.MalformedError('The private_key field was not found in the service account info.')
    return cls.from_string(info[_JSON_FILE_PRIVATE_KEY], info.get(_JSON_FILE_PRIVATE_KEY_ID))