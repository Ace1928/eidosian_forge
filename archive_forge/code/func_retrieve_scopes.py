import logging
import warnings
from six.moves import http_client
from oauth2client import client
from oauth2client.contrib import _metadata
def retrieve_scopes(self, http):
    """Retrieves the canonical list of scopes for this access token.

        Overrides client.Credentials.retrieve_scopes. Fetches scopes info
        from the metadata server.

        Args:
            http: httplib2.Http, an http object to be used to make the refresh
                  request.

        Returns:
            A set of strings containing the canonical list of scopes.
        """
    self._retrieve_info(http)
    return self.scopes