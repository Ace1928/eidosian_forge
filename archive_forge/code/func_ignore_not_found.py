import abc
import weakref
from keystoneauth1 import exceptions
from keystoneauth1.identity import generic
from keystoneauth1 import plugin
from oslo_config import cfg
from oslo_utils import excutils
import requests
from heat.common import config
from heat.common import exception as heat_exception
@excutils.exception_filter
def ignore_not_found(self, ex):
    """Raises the exception unless it is a not-found."""
    return self.is_not_found(ex)