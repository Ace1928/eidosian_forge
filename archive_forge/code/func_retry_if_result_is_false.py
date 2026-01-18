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
def retry_if_result_is_false(result):
    return result is False