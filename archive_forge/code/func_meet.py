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
def meet(self, timeout_s=180):
    """Meet at the named actor store.

        Args:
            timeout_s: timeout in seconds.

        Return:
            None
        """
    if timeout_s <= 0:
        raise ValueError("The 'timeout' argument must be positive. Got '{}'.".format(timeout_s))
    self._store_name = get_store_name(self._store_key)
    timeout_delta = datetime.timedelta(seconds=timeout_s)
    elapsed = datetime.timedelta(seconds=0)
    start_time = datetime.datetime.now()
    while elapsed < timeout_delta:
        try:
            logger.debug("Trying to meet at the store '{}'".format(self._store_name))
            self._store = ray.get_actor(self._store_name)
        except ValueError:
            logger.debug("Failed to meet at the store '{}'.Trying again...".format(self._store_name))
            time.sleep(1)
            elapsed = datetime.datetime.now() - start_time
            continue
        logger.debug('Successful rendezvous!')
        break
    if not self._store:
        raise RuntimeError('Unable to meet other processes at the rendezvous store. If you are using P2P communication, please check if tensors are put in the correct GPU. ')