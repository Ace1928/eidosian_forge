import datetime
from typing import Dict, Optional, Union
import pytest
import cirq
import cirq_google
import sympy
import numpy as np
from cirq_google.api import v2
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program_test import NothingProgram
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
def test_no_calibration_from_job():
    proc = SimulatedLocalProcessor(processor_id='test_proc')
    engine = SimulatedLocalEngine([proc])
    job = engine.get_processor('test_proc').run_sweep(cirq.Circuit(), params={}, repetitions=100)
    assert job.get_processor() == proc
    assert job.get_calibration() is None