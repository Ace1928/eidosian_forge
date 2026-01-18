import io
import logging
import threading
import traceback
from typing import Any
import google.protobuf.message
import ray._private.utils
import ray.cloudpickle as pickle
from ray._private import ray_constants
from ray._raylet import (
from ray.core.generated.common_pb2 import ErrorType, RayErrorInfo
from ray.exceptions import (
from ray.util import serialization_addons
from ray.util import inspect_serializability
def object_ref_reducer(obj):
    worker = ray._private.worker.global_worker
    worker.check_connected()
    self.add_contained_object_ref(obj)
    obj, owner_address, object_status = worker.core_worker.serialize_object_ref(obj)
    return (_object_ref_deserializer, (obj.binary(), obj.call_site(), owner_address, object_status))