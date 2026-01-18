import uuid
from typing import Generic, TypeVar, Optional
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from .measurements import MeasurementProcess, MidMeasure
def merged_fn(*x):
    sub_args_1 = (x[i] for i in [merged_measurements.index(m) for m in self.measurements])
    sub_args_2 = (x[i] for i in [merged_measurements.index(m) for m in other.measurements])
    out_1 = self.processing_fn(*sub_args_1)
    out_2 = other.processing_fn(*sub_args_2)
    return (out_1, out_2)