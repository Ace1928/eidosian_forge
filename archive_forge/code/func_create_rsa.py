import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def create_rsa(self, name=None, public_key=None, private_key=None, private_key_passphrase=None):
    """Factory method for `RSAContainer` objects

        `RSAContainer` objects returned by this method have not yet been
        stored in Barbican.

        :param name: A friendly name for the RSAContainer
        :param public_key: Secret object containing a Public Key
        :param private_key: Secret object containing a Private Key
        :param private_key_passphrase: Secret object containing a passphrase
        :returns: RSAContainer
        :rtype: :class:`barbicanclient.v1.containers.RSAContainer`
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
    return RSAContainer(api=self._api, name=name, public_key=public_key, private_key=private_key, private_key_passphrase=private_key_passphrase)