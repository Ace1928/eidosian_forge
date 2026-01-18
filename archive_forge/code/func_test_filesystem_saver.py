import os
import pytest
import cirq
import cirq_google as cg
import numpy as np
from cirq_google.workflow.io import _FilesystemSaver
def test_filesystem_saver(tmpdir) -> None:
    run_id = 'asdf'
    fs_saver = _FilesystemSaver(base_data_dir=tmpdir, run_id=run_id)
    rt_config = cg.QuantumRuntimeConfiguration(processor_record=cg.SimulatedProcessorWithLocalDeviceRecord('rainbow'), run_id=run_id)
    shared_rt_info = cg.SharedRuntimeInfo(run_id=run_id)
    fs_saver.initialize(rt_config, shared_rt_info=shared_rt_info)
    rt_config2 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/QuantumRuntimeConfiguration.json.gz')
    shared_rt_info2 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/SharedRuntimeInfo.json.gz')
    assert rt_config == rt_config2
    assert shared_rt_info == shared_rt_info2
    shared_rt_info.run_id = 'updated_run_id'
    exe_result = cg.ExecutableResult(spec=None, runtime_info=cg.RuntimeInfo(execution_index=0), raw_data=cirq.ResultDict(params=cirq.ParamResolver({}), measurements={'z': np.ones((100, 5))}))
    fs_saver.consume_result(exe_result=exe_result, shared_rt_info=shared_rt_info)
    shared_rt_info3 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/SharedRuntimeInfo.json.gz')
    exe_result3 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/ExecutableResult.0.json.gz')
    assert shared_rt_info == shared_rt_info3
    assert exe_result == exe_result3
    egr_record: cg.ExecutableGroupResultFilesystemRecord = cirq.read_json_gzip(f'{fs_saver.data_dir}/ExecutableGroupResultFilesystemRecord.json.gz')
    assert egr_record == fs_saver.egr_record
    exegroup_result: cg.ExecutableGroupResult = egr_record.load(base_data_dir=tmpdir)
    assert exegroup_result.shared_runtime_info == shared_rt_info
    assert exegroup_result.runtime_configuration == rt_config
    assert exegroup_result.executable_results[0] == exe_result