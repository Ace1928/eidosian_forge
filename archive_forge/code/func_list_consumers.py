import base64
import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import exceptions
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
def list_consumers(self, secret_ref, limit=10, offset=0):
    """List consumers of the secret

        :param secret_ref: Full HATEOAS reference to a secret, or a UUID
        :param limit: Max number of consumers returned
        :param offset: Offset secrets to begin list
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
    LOG.debug('Listing consumers of secret {0}'.format(secret_ref))
    self._enforce_microversion()
    secret_uuid = base.validate_ref_and_return_uuid(secret_ref, 'secret')
    href = '{0}/{1}/consumers'.format(self._entity, secret_uuid)
    params = {'limit': limit, 'offset': offset}
    response = self._api.get(href, params=params)
    return [SecretConsumers(secret_ref=secret_ref, **s) for s in response.get('consumers', [])]