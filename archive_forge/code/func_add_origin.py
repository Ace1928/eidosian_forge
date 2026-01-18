import copy
import logging
import debtcollector
from oslo_config import cfg
from oslo_middleware import base
import webob.exc
def add_origin(self, allowed_origin, allow_credentials=True, expose_headers=None, max_age=None, allow_methods=None, allow_headers=None):
    """Add another origin to this filter.

        :param allowed_origin: Protocol, host, and port for the allowed origin.
        :param allow_credentials: Whether to permit credentials.
        :param expose_headers: A list of headers to expose.
        :param max_age: Maximum cache duration.
        :param allow_methods: List of HTTP methods to permit.
        :param allow_headers: List of HTTP headers to permit from the client.
        :return:
        """
    if isinstance(allowed_origin, str):
        LOG.warning('DEPRECATED: The `allowed_origin` keyword argument in `add_origin()` should be a list, found String.')
        allowed_origin = [allowed_origin]
    if allowed_origin:
        for origin in allowed_origin:
            if origin in self.allowed_origins:
                LOG.warning('Allowed origin [%s] already exists, skipping' % (allowed_origin,))
                continue
            self.allowed_origins[origin] = {'allow_credentials': allow_credentials, 'expose_headers': expose_headers, 'max_age': max_age, 'allow_methods': allow_methods, 'allow_headers': allow_headers}