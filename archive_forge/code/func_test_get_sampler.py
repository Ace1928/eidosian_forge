import datetime
from typing import Dict, List, Optional, Union
import pytest
import cirq
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program_test import NothingProgram
from cirq_google.engine.abstract_local_engine import AbstractLocalEngine
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_program import AbstractProgram
import cirq_google.engine.calibration as calibration
def test_get_sampler():
    processor = ProgramDictProcessor(programs={}, processor_id='grocery')
    engine = NothingEngine([processor])
    assert isinstance(engine.get_sampler('grocery'), cirq.Sampler)
    with pytest.raises(ValueError, match='Invalid processor'):
        engine.get_sampler(['blah'])