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
def test_executable_group_result(tmpdir):
    egr = cg.ExecutableGroupResult(runtime_configuration=cg.QuantumRuntimeConfiguration(processor_record=cg.SimulatedProcessorWithLocalDeviceRecord('rainbow'), run_id='unit-test'), shared_runtime_info=cg.SharedRuntimeInfo(run_id='my run'), executable_results=[cg.ExecutableResult(spec=_get_example_spec(name=f'test-spec-{i}'), runtime_info=cg.RuntimeInfo(execution_index=i), raw_data=cirq.ResultDict(params=cirq.ParamResolver(), measurements={'z': np.ones((1000, 4))})) for i in range(3)])
    cg_assert_equivalent_repr(egr)
    assert len(egr.executable_results) == 3
    _assert_json_roundtrip(egr, tmpdir)