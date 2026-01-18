from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneclient import access
from keystoneclient import base
from keystoneclient.i18n import _
def get_token_data(self, token):
    """Fetch the data about a token from the identity server.

        :param str token: The token id.

        :rtype: dict
        """
    url = '/tokens/%s' % token
    resp, body = self.client.get(url)
    return body