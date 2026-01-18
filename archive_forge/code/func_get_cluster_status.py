import time
from collections import defaultdict
from typing import List
from ray._raylet import GcsClient
from ray.autoscaler.v2.schema import ClusterStatus, Stats
from ray.autoscaler.v2.utils import ClusterStatusParser
from ray.core.generated.autoscaler_pb2 import GetClusterStatusReply
def get_cluster_status(gcs_address: str, timeout: int=DEFAULT_RPC_TIMEOUT_S) -> ClusterStatus:
    """
    Get the cluster status from the autoscaler.

    Args:
        gcs_address: The GCS address to query.
        timeout: Timeout in seconds for the request to be timeout

    Returns:
        A ClusterStatus object.
    """
    assert len(gcs_address) > 0, 'GCS address is not specified.'
    req_time = time.time()
    str_reply = GcsClient(gcs_address).get_cluster_status(timeout_s=timeout)
    reply_time = time.time()
    reply = GetClusterStatusReply()
    reply.ParseFromString(str_reply)
    return ClusterStatusParser.from_get_cluster_status_reply(reply, stats=Stats(gcs_request_time_s=reply_time - req_time, request_ts_s=req_time))