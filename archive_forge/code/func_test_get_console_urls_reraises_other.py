import collections
from unittest import mock
import uuid
from novaclient import client as nc
from novaclient import exceptions as nova_exceptions
from oslo_config import cfg
from oslo_serialization import jsonutils as json
import requests
from heat.common import exception
from heat.engine.clients.os import nova
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_get_console_urls_reraises_other(self):
    exc = Exception
    self.console_method.side_effect = exc('spam')
    self._test_get_console_url_tolerate_exception('spam')