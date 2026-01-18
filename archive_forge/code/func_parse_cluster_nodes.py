import datetime
from redis.utils import str_if_bytes
def parse_cluster_nodes(response, **options):
    """
    @see: https://redis.io/commands/cluster-nodes  # string / bytes
    @see: https://redis.io/commands/cluster-replicas # list of string / bytes
    """
    if isinstance(response, (str, bytes)):
        response = response.splitlines()
    return dict((_parse_node_line(str_if_bytes(node)) for node in response))