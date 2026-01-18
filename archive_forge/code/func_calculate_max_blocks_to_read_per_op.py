from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict
def calculate_max_blocks_to_read_per_op(self, topology: 'Topology') -> Dict['OpState', int]:
    """Determine how many blocks of data we can read from each operator.
        The `DataOpTask`s of the operators will stop reading blocks when the limit is
        reached. Then the execution of these tasks will be paused when the streaming
        generator backpressure threshold is reached.
        Used in `streaming_executor_state.py::process_completed_tasks()`.

        Returns: A dict mapping from each operator's OpState to the desired number of
            blocks to read. For operators that are not in the dict, all available blocks
            will be read.

        Note: Only one backpressure policy that implements this method can be enabled
            at a time.
        """
    return {}