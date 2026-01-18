import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING
import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
def consume_result(self, exe_result: 'cg.ExecutableResult', shared_rt_info: 'cg.SharedRuntimeInfo'):
    """Save an `cg.ExecutableResult` that has been completed.

        Args:
            exe_result: The completed `cg.ExecutableResult` to be saved to
                'ExecutableResult.{i}.json.gz'
            shared_rt_info: The current `cg.SharedRuntimeInfo` to update SharedRuntimeInfo.json.gz.
        """
    i = exe_result.runtime_info.execution_index
    exe_result_path = f'ExecutableResult.{i}.json.gz'
    cirq.to_json_gzip(exe_result, f'{self._data_dir}/{exe_result_path}')
    self._egr_record.executable_result_paths.append(exe_result_path)
    _update_updatable_files(self._egr_record, shared_rt_info, self._data_dir)