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
def create_noiseless_virtual_engine_from_latest_templates() -> SimulatedLocalEngine:
    """Creates a noiseless virtual engine based on current templates.

    This uses the most recent templates to create a reasonable facsimile of
    a simulated Quantum Computing Service (QCS).

    Note:  this will use the most recent templates to match the service.
    While not expected to change frequently, this function may change the
    templates (processors) that are included in the "service" as the actual
    hardware evolves.  The processors returned from this function should not
    be considered stable from version to version and are not guaranteed to be
    backwards compatible.
    """
    processor_ids = list(MOST_RECENT_TEMPLATES.keys())
    template_names = [MOST_RECENT_TEMPLATES[k] for k in processor_ids]
    return create_noiseless_virtual_engine_from_templates(processor_ids, template_names)