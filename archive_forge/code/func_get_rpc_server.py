import eventlet
from oslo_config import cfg
import oslo_messaging
from oslo_messaging.rpc import dispatcher
from osprofiler import profiler
from heat.common import context
def get_rpc_server(target, endpoint):
    """Return a configured oslo_messaging rpc server."""
    serializer = RequestContextSerializer(oslo_messaging.JsonPayloadSerializer())
    access_policy = dispatcher.DefaultRPCAccessPolicy
    return oslo_messaging.get_rpc_server(TRANSPORT, target, [endpoint], executor='eventlet', serializer=serializer, access_policy=access_policy)