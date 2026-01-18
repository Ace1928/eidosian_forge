import logging
import datetime
import time
import ray
import cupy
from ray.util.collective.const import ENV
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
from ray.util.collective.collective_group.cuda_stream import get_stream_pool
def get_nccl_id(self, timeout_s=180):
    """Get the NCCLUniqueID from the store through Ray.

        Args:
            timeout_s: timeout in seconds.

        Return:
            uid: the NCCLUniqueID if successful.
        """
    if not self._store:
        raise ValueError('Rendezvous store is not setup.')
    uid = None
    timeout_delta = datetime.timedelta(seconds=timeout_s)
    elapsed = datetime.timedelta(seconds=0)
    start_time = datetime.datetime.now()
    while elapsed < timeout_delta:
        uid = ray.get(self._store.get_id.remote())
        if not uid:
            time.sleep(1)
            elapsed = datetime.datetime.now() - start_time
            continue
        break
    if not uid:
        raise RuntimeError('Unable to get the NCCLUniqueID from the store.')
    return uid