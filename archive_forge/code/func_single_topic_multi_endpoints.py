import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
def single_topic_multi_endpoints(scenario):
    params = scenario[1]
    single_exchange = params['exchange1'] == params['exchange2']
    single_topic = params['topic1'] == params['topic2']
    if single_topic and single_exchange and params['multi_endpoints']:
        params['expect_either'] = params['expect1'] + params['expect2']
        params['expect1'] = params['expect2'] = []
    else:
        params['expect_either'] = []
    return scenario