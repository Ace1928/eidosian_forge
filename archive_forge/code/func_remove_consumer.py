import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def remove_consumer(self, container_ref, name, url):
    """Remove a consumer from the container

        :param container_ref: Full HATEOAS reference to a Container, or a UUID
        :param name: Name of the previously consuming service
        :param url: URL of the previously consuming resource
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
    LOG.debug('Deleting consumer registration for container {0} as {1}: {2}'.format(container_ref, name, url))
    container_uuid = base.validate_ref_and_return_uuid(container_ref, 'Container')
    href = '{0}/{1}/consumers'.format(self._entity, container_uuid)
    consumer_dict = {'name': name, 'URL': url}
    self._api.delete(href, json=consumer_dict)