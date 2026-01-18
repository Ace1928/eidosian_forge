import functools
import logging
from oslo_utils.timeutils import parse_isotime
from barbicanclient import base
from barbicanclient import formatter
from barbicanclient.v1 import acls as acl_manager
from barbicanclient.v1 import secrets as secret_manager
def register_consumer(self, container_ref, name, url):
    """Add a consumer to the container

        :param container_ref: Full HATEOAS reference to a Container, or a UUID
        :param name: Name of the consuming service
        :param url: URL of the consuming resource
        :returns: A container object per the get() method
        :raises barbicanclient.exceptions.HTTPAuthError: 401 Responses
        :raises barbicanclient.exceptions.HTTPClientError: 4xx Responses
        :raises barbicanclient.exceptions.HTTPServerError: 5xx Responses
        """
    LOG.debug('Creating consumer registration for container {0} as {1}: {2}'.format(container_ref, name, url))
    container_uuid = base.validate_ref_and_return_uuid(container_ref, 'Container')
    href = '{0}/{1}/consumers'.format(self._entity, container_uuid)
    consumer_dict = dict()
    consumer_dict['name'] = name
    consumer_dict['URL'] = url
    response = self._api.post(href, json=consumer_dict)
    return self._generate_typed_container(response)