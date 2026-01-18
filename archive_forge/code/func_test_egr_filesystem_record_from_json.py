import os
import pytest
import cirq
import cirq_google as cg
import numpy as np
from cirq_google.workflow.io import _FilesystemSaver
def test_egr_filesystem_record_from_json(tmpdir):
    run_id = 'my-run-id'
    egr_fs_record = cg.ExecutableGroupResultFilesystemRecord(runtime_configuration_path='RuntimeConfiguration.json.gz', shared_runtime_info_path='SharedRuntimeInfo.jzon.gz', executable_result_paths=['ExecutableResult.1.json.gz', 'ExecutableResult.2.json.gz'], run_id=run_id)
    os.makedirs(f'{tmpdir}/{run_id}')
    cirq.to_json_gzip(egr_fs_record, f'{tmpdir}/{run_id}/ExecutableGroupResultFilesystemRecord.json.gz')
    egr_fs_record2 = cg.ExecutableGroupResultFilesystemRecord.from_json(run_id=run_id, base_data_dir=tmpdir)
    assert egr_fs_record == egr_fs_record2
    cirq.to_json_gzip(cirq.Circuit(), f'{tmpdir}/{run_id}/ExecutableGroupResultFilesystemRecord.json.gz')
    with pytest.raises(ValueError, match='.*not an `ExecutableGroupFilesystemRecord`.'):
        cg.ExecutableGroupResultFilesystemRecord.from_json(run_id=run_id, base_data_dir=tmpdir)
    os.makedirs(f'{tmpdir}/questionable_run_id')
    cirq.to_json_gzip(egr_fs_record, f'{tmpdir}/questionable_run_id/ExecutableGroupResultFilesystemRecord.json.gz')
    with pytest.raises(ValueError, match='.*does not match the provided run_id'):
        cg.ExecutableGroupResultFilesystemRecord.from_json(run_id='questionable_run_id', base_data_dir=tmpdir)