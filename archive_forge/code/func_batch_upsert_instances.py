import copy
import logging
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from ray.autoscaler.v2.instance_manager.storage import Storage, StoreStatus
from ray.core.generated.instance_manager_pb2 import Instance
def batch_upsert_instances(self, updates: List[Instance], expected_storage_version: Optional[int]=None) -> StoreStatus:
    """Upsert instances into the storage. If the instance already exists,
        it will be updated. Otherwise, it will be inserted. If the
        expected_storage_version is specified, the update will fail if the
        current storage version does not match the expected version.

        Note the version of the upserted instances will be set to the current
        storage version.

        Args:
            updates: A list of instances to be upserted.
            expected_storage_version: The expected storage version.

        Returns:
            StoreStatus: A tuple of (success, storage_version).
        """
    mutations = {}
    version = self._storage.get_version()
    if expected_storage_version and expected_storage_version != version:
        return StoreStatus(False, version)
    for instance in updates:
        instance = copy.deepcopy(instance)
        instance.version = 0
        instance.timestamp_since_last_modified = int(time.time())
        mutations[instance.instance_id] = instance.SerializeToString()
    result, version = self._storage.batch_update(self._table_name, mutations, {}, expected_storage_version)
    if result:
        for subscriber in self._status_change_subscribers:
            subscriber.notify([InstanceUpdateEvent(instance_id=instance.instance_id, new_status=instance.status, new_ray_status=instance.ray_status) for instance in updates])
    return StoreStatus(result, version)