from __future__ import absolute_import
import collections
import functools
import logging
from ._collections import RecentlyUsedContainer
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, port_by_scheme
from .exceptions import (
from .packages import six
from .packages.six.moves.urllib.parse import urljoin
from .request import RequestMethods
from .util.proxy import connection_requires_http_tunnel
from .util.retry import Retry
from .util.url import parse_url
def connection_from_pool_key(self, pool_key, request_context=None):
    """
        Get a :class:`urllib3.connectionpool.ConnectionPool` based on the provided pool key.

        ``pool_key`` should be a namedtuple that only contains immutable
        objects. At a minimum it must have the ``scheme``, ``host``, and
        ``port`` fields.
        """
    with self.pools.lock:
        pool = self.pools.get(pool_key)
        if pool:
            return pool
        scheme = request_context['scheme']
        host = request_context['host']
        port = request_context['port']
        pool = self._new_pool(scheme, host, port, request_context=request_context)
        self.pools[pool_key] = pool
    return pool