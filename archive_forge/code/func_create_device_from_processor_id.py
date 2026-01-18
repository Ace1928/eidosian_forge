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
def create_device_from_processor_id(processor_id: str) -> cirq.Device:
    """Generates a `cirq.Device` for a given processor ID.

    Args:
        processor_id: name of the processor to simulate.

    Raises:
        ValueError: if processor_id is not a supported QCS processor.
    """
    device_specification = create_device_spec_from_processor_id(processor_id)
    return grid_device.GridDevice.from_proto(device_specification)