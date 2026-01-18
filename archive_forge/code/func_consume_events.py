import kombu
from oslo_config import cfg
from oslo_messaging._drivers import common
from oslo_messaging import transport
import requests
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def consume_events(self, handler, count):
    self.conn.drain_events()
    return len(handler.notifications) == count