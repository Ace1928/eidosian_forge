import eventlet
from oslo_config import cfg
import oslo_messaging
from oslo_messaging.rpc import dispatcher
from osprofiler import profiler
from heat.common import context
def get_rpc_client(**kwargs):
    """Return a configured oslo_messaging RPCClient."""
    target = oslo_messaging.Target(**kwargs)
    serializer = RequestContextSerializer(oslo_messaging.JsonPayloadSerializer())
    return oslo_messaging.get_rpc_client(TRANSPORT, target, serializer=serializer)