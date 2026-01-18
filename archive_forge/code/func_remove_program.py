import datetime
from typing import Dict, List, Optional, Sequence, Union
import cirq
from cirq_google.api import v2
from cirq_google.engine import calibration, validating_sampler
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.local_simulation_type import LocalSimulationType
from cirq_google.engine.simulated_local_job import SimulatedLocalJob
from cirq_google.engine.simulated_local_program import SimulatedLocalProgram
from cirq_google.serialization.circuit_serializer import CIRCUIT_SERIALIZER
from cirq_google.engine.processor_sampler import ProcessorSampler
from cirq_google.engine import engine_validator
def remove_program(self, program_id: str):
    """Remove reference to a child program."""
    if program_id in self._programs:
        del self._programs[program_id]