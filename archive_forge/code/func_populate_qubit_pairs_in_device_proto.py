from typing import Collection, Dict, Optional, List, Set, Tuple, cast
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import device_pb2
from cirq_google.devices import grid_device
from cirq_google.experimental.ops import coupler_pulse
from cirq_google.ops import physical_z_tag, sycamore_gate
def populate_qubit_pairs_in_device_proto(pairs: Collection[Tuple[cirq.Qid, cirq.Qid]], out: device_pb2.DeviceSpecification) -> None:
    """Populates `DeviceSpecification.valid_targets` with the device's qubit pairs.

    Args:
        pairs: The collection of the device's bi-directional qubit pairs.
        out: The `DeviceSpecification` to be populated.
    """
    grid_targets = out.valid_targets.add()
    grid_targets.name = _2_QUBIT_TARGET_SET
    grid_targets.target_ordering = device_pb2.TargetSet.SYMMETRIC
    for pair in pairs:
        new_target = grid_targets.targets.add()
        new_target.ids.extend((v2.qubit_to_proto_id(q) for q in pair))