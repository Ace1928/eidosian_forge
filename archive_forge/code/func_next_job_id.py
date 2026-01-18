import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def next_job_id(self):
    """Get next job id from GCS.

        Returns:
            Next job id in the cluster.
        """
    self._check_connected()
    return ray.JobID.from_int(self.global_state_accessor.get_next_job_id())