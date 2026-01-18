import eventlet
from oslo_config import cfg
import oslo_messaging
from oslo_messaging.rpc import dispatcher
from osprofiler import profiler
from heat.common import context
def setup_transports(url, optional):
    global TRANSPORT, NOTIFICATIONS_TRANSPORT
    oslo_messaging.set_transport_defaults('heat')
    exmods = ['heat.common.exception']
    TRANSPORT = get_specific_transport(url, optional, exmods)
    NOTIFICATIONS_TRANSPORT = get_specific_transport(url, optional, exmods, is_for_notifications=True)