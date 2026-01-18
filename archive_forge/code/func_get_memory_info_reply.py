import ray
import ray._private.profiling as profiling
import ray._private.services as services
import ray._private.utils as utils
import ray._private.worker
from ray._private import ray_constants
from ray._private.state import GlobalState
from ray._raylet import GcsClientOptions
def get_memory_info_reply(state, node_manager_address=None, node_manager_port=None):
    """Returns global memory info."""
    from ray.core.generated import node_manager_pb2, node_manager_pb2_grpc
    if node_manager_address is None or node_manager_port is None:
        raylet = None
        for node in state.node_table():
            if node['Alive']:
                raylet = node
                break
        assert raylet is not None, 'Every raylet is dead'
        raylet_address = '{}:{}'.format(raylet['NodeManagerAddress'], raylet['NodeManagerPort'])
    else:
        raylet_address = '{}:{}'.format(node_manager_address, node_manager_port)
    channel = utils.init_grpc_channel(raylet_address, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    stub = node_manager_pb2_grpc.NodeManagerServiceStub(channel)
    reply = stub.FormatGlobalMemoryInfo(node_manager_pb2.FormatGlobalMemoryInfoRequest(include_memory_info=False), timeout=60.0)
    return reply