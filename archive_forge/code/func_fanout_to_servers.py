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
def fanout_to_servers(scenario):
    params = scenario[1]
    fanout = params['fanout1'] or params['fanout2']
    single_exchange = params['exchange1'] == params['exchange2']
    single_topic = params['topic1'] == params['topic2']
    multi_servers = params['server1'] != params['server2']
    if fanout and single_exchange and single_topic and multi_servers:
        params['expect1'] = params['expect1'][:] + params['expect1']
        params['expect2'] = params['expect2'][:] + params['expect2']
    return scenario