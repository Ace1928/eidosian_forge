import uuid
from oslo_utils import timeutils
from heat.rpc import listener_client
def engine_alive(context, engine_id):
    return listener_client.EngineListenerClient(engine_id).is_alive(context)