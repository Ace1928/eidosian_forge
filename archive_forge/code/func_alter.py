from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.typing import KeyT, Number
def alter(self, key: KeyT, retention_msecs: Optional[int]=None, labels: Optional[Dict[str, str]]=None, chunk_size: Optional[int]=None, duplicate_policy: Optional[str]=None):
    """
        Update the retention, chunk size, duplicate policy, and labels of an existing
        time series.

        Args:

        key:
            time-series key
        retention_msecs:
            Maximum retention period, compared to maximal existing timestamp (in milliseconds).
            If None or 0 is passed then  the series is not trimmed at all.
        labels:
            Set of label-value pairs that represent metadata labels of the key.
        chunk_size:
            Memory size, in bytes, allocated for each data chunk.
            Must be a multiple of 8 in the range [128 .. 1048576].
        duplicate_policy:
            Policy for handling multiple samples with identical timestamps.
            Can be one of:
            - 'block': an error will occur for any out of order sample.
            - 'first': ignore the new value.
            - 'last': override with latest value.
            - 'min': only override if the value is lower than the existing value.
            - 'max': only override if the value is higher than the existing value.
            - 'sum': If a previous sample exists, add the new sample to it so that             the updated value is equal to (previous + new). If no previous sample             exists, set the updated value equal to the new value.

        For more information: https://redis.io/commands/ts.alter/
        """
    params = [key]
    self._append_retention(params, retention_msecs)
    self._append_chunk_size(params, chunk_size)
    self._append_duplicate_policy(params, ALTER_CMD, duplicate_policy)
    self._append_labels(params, labels)
    return self.execute_command(ALTER_CMD, *params)