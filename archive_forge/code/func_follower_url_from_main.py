from __future__ import annotations
import collections
import logging
from . import config
from . import engines
from . import util
from .. import exc
from .. import inspect
from ..engine import url as sa_url
from ..sql import ddl
from ..sql import schema
@register.init
def follower_url_from_main(url, ident):
    """Create a connection URL for a dynamically-created test database.

    :param url: the connection URL specified when the test run was invoked
    :param ident: the pytest-xdist "worker identifier" to be used as the
                  database name
    """
    url = sa_url.make_url(url)
    return url.set(database=ident)