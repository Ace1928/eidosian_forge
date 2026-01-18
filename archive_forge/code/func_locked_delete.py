import threading
import keyring
from oauth2client import client
def locked_delete(self):
    """Delete Credentials file.

        Args:
            credentials: Credentials, the credentials to store.
        """
    keyring.set_password(self._service_name, self._user_name, '')