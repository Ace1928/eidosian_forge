import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
def export_update(self) -> Optional[model_update_pb2.ModelUpdateProto]:
    """Returns changes to the model since last call to checkpoint/creation."""
    return self.storage_update_tracker.export_update()