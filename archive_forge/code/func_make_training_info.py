import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_training_info(algorithm: GraphProto, algorithm_bindings: AssignmentBindingType, initialization: Optional[GraphProto], initialization_bindings: Optional[AssignmentBindingType]) -> TrainingInfoProto:
    training_info = TrainingInfoProto()
    training_info.algorithm.CopyFrom(algorithm)
    for k, v in algorithm_bindings:
        binding = training_info.update_binding.add()
        binding.key = k
        binding.value = v
    if initialization:
        training_info.initialization.CopyFrom(initialization)
    if initialization_bindings:
        for k, v in initialization_bindings:
            binding = training_info.initialization_binding.add()
            binding.key = k
            binding.value = v
    return training_info