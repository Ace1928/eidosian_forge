import json
from typing import cast, List, Optional, Union, Type
import pathlib
import time
import google.protobuf.text_format as text_format
import cirq
from cirq.sim.simulator import SimulatesSamples
from cirq_google.api import v2
from cirq_google.engine import calibration, engine_validator, simulated_local_processor, util
from cirq_google.devices import grid_device
from cirq_google.devices.google_noise_properties import NoiseModelFromGoogleNoiseProperties
from cirq_google.engine.calibration_to_noise_properties import noise_properties_from_calibration
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
def create_noiseless_virtual_processor_from_proto(processor_id: str, device_specification: v2.device_pb2.DeviceSpecification) -> SimulatedLocalProcessor:
    """Creates a simulated local processor from a device specification proto.

    The device specification protocol buffer specifies qubits and gates on the device
    and can be retrieved from a stored "proto.txt" file or from the QCS API.

    Args:
        processor_id: name of the processor to simulate.  This is an arbitrary
            string identifier and does not have to match the processor's name
            in QCS.
        device_specification:  `v2.device_pb2.DeviceSpecification` proto to create
            a validating device from.
        gate_sets: Iterable of serializers to use in the processor.
    """
    device = grid_device.GridDevice.from_proto(device_specification)
    processor = _create_virtual_processor_from_device(processor_id, device, device_specification)
    return processor