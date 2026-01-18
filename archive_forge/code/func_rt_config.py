import datetime
import glob
import re
import time
import uuid
from typing import List, cast, Any
import numpy as np
import pytest
import cirq
import cirq_google as cg
from cirq_google.workflow.quantum_executable_test import _get_quantum_executables, _get_example_spec
from cirq_google.workflow.quantum_runtime import _time_into_runtime_info
@pytest.fixture(params=['minimal', 'full'])
def rt_config(request):
    if request.param == 'minimal':
        return cg.QuantumRuntimeConfiguration(processor_record=cg.SimulatedProcessorWithLocalDeviceRecord('rainbow'))
    elif request.param == 'full':
        return cg.QuantumRuntimeConfiguration(processor_record=cg.SimulatedProcessorWithLocalDeviceRecord('rainbow'), run_id='unit-test', random_seed=52, qubit_placer=cg.RandomDevicePlacer(), target_gateset=cirq.CZTargetGateset())
    raise ValueError(f'Unknown flavor {request}')