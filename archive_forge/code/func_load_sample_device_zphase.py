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
def load_sample_device_zphase(processor_id: str) -> util.ZPhaseDataType:
    """Loads sample Z phase errors for the given device.

    Args:
        processor_id: name of the processor to simulate.

    Returns:
        Z phases in the form {gate_type: {angle_type: {qubit_pair: error}}},
        where gate_type is "syc" or "sqrt_iswap", angle_type is "zeta" or
        "gamma", and "qubit_pair" is a tuple of qubits.

    Raises:
        ValueError: if processor_id is not a supported QCS processor.
    """
    zphase_name = ZPHASE_DATA.get(processor_id, None)
    if zphase_name is None:
        raise ValueError(f'Got processor_id={processor_id}, but no Z phase data is defined for that processor.')
    path = pathlib.Path(__file__).parent.parent.resolve()
    with path.joinpath('devices', 'calibrations', zphase_name).open() as f:
        raw_data = json.load(f)
        nested_data: util.ZPhaseDataType = {gate_type: {angle: {(v2.qubit_from_proto_id(q0), v2.qubit_from_proto_id(q1)): vals for q0, q1, vals in triples} for angle, triples in angles.items()} for gate_type, angles in raw_data.items()}
    return nested_data