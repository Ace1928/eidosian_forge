import os
import pytest
import cirq
import cirq_google as cg
import numpy as np
from cirq_google.workflow.io import _FilesystemSaver
def test_egr_filesystem_record_repr():
    egr_fs_record = cg.ExecutableGroupResultFilesystemRecord(runtime_configuration_path='RuntimeConfiguration.json.gz', shared_runtime_info_path='SharedRuntimeInfo.jzon.gz', executable_result_paths=['ExecutableResult.1.json.gz', 'ExecutableResult.2.json.gz'], run_id='my-run-id')
    cg_assert_equivalent_repr(egr_fs_record)