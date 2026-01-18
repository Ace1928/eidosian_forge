import uuid
from oslo_utils import timeutils
from heat.rpc import listener_client
def generate_engine_id():
    return str(uuid.uuid4())