import base64
import logging
from collections import defaultdict
from enum import Enum
from typing import List
import ray
from ray._private.internal_api import node_stats
from ray._raylet import ActorID, JobID, TaskID
def decode_object_ref_if_needed(object_ref: str) -> bytes:
    """Decode objectRef bytes string.

    gRPC reply contains an objectRef that is encodded by Base64.
    This function is used to decode the objectRef.
    Note that there are times that objectRef is already decoded as
    a hex string. In this case, just convert it to a binary number.
    """
    if object_ref.endswith('='):
        return base64.standard_b64decode(object_ref)
    else:
        return ray._private.utils.hex_to_binary(object_ref)