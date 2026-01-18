import collections
from functools import partial
import itertools
import sys
from numbers import Number
from typing import Dict, Iterator, Set, Union
from typing import List, Optional
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI, ExperimentalAPI, PublicAPI
from ray.rllib.utils.compression import pack, unpack, is_compressed
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
from ray.util import log_once
@PublicAPI
def split_by_episode(self, key: Optional[str]=None) -> List['SampleBatch']:
    """Splits by `eps_id` column and returns list of new batches.
        If `eps_id` is not present, splits by `dones` instead.

        Args:
            key: If specified, overwrite default and use key to split.

        Returns:
            List of batches, one per distinct episode.

        Raises:
            KeyError: If the `eps_id` AND `dones` columns are not present.

        .. testcode::
            :skipif: True

            from ray.rllib.policy.sample_batch import SampleBatch
            # "eps_id" is present
            batch = SampleBatch(
                {"a": [1, 2, 3], "eps_id": [0, 0, 1]})
            print(batch.split_by_episode())

            # "eps_id" not present, split by "dones" instead
            batch = SampleBatch(
                {"a": [1, 2, 3, 4, 5], "dones": [0, 0, 1, 0, 1]})
            print(batch.split_by_episode())

            # The last episode is appended even if it does not end with done
            batch = SampleBatch(
                {"a": [1, 2, 3, 4, 5], "dones": [0, 0, 1, 0, 0]})
            print(batch.split_by_episode())

            batch = SampleBatch(
                {"a": [1, 2, 3, 4, 5], "dones": [0, 0, 0, 0, 0]})
            print(batch.split_by_episode())


        .. testoutput::

            [{"a": [1, 2], "eps_id": [0, 0]}, {"a": [3], "eps_id": [1]}]
            [{"a": [1, 2, 3], "dones": [0, 0, 1]}, {"a": [4, 5], "dones": [0, 1]}]
            [{"a": [1, 2, 3], "dones": [0, 0, 1]}, {"a": [4, 5], "dones": [0, 0]}]
            [{"a": [1, 2, 3, 4, 5], "dones": [0, 0, 0, 0, 0]}]


        """
    assert key is None or key in [SampleBatch.EPS_ID, SampleBatch.DONES], f"`SampleBatch.split_by_episode(key={key})` invalid! Must be [None|'dones'|'eps_id']."

    def slice_by_eps_id():
        slices = []
        cur_eps_id = self[SampleBatch.EPS_ID][0]
        offset = 0
        for i in range(self.count):
            next_eps_id = self[SampleBatch.EPS_ID][i]
            if next_eps_id != cur_eps_id:
                slices.append(self[offset:i])
                offset = i
                cur_eps_id = next_eps_id
        slices.append(self[offset:self.count])
        return slices

    def slice_by_terminateds_or_truncateds():
        slices = []
        offset = 0
        for i in range(self.count):
            if self[SampleBatch.TERMINATEDS][i] or (SampleBatch.TRUNCATEDS in self and self[SampleBatch.TRUNCATEDS][i]):
                slices.append(self[offset:i + 1])
                offset = i + 1
        if offset != self.count:
            slices.append(self[offset:])
        return slices
    key_to_method = {SampleBatch.EPS_ID: slice_by_eps_id, SampleBatch.DONES: slice_by_terminateds_or_truncateds}
    key_resolve_order = [SampleBatch.EPS_ID, SampleBatch.DONES]
    slices = None
    if key is not None:
        if key == SampleBatch.EPS_ID and key not in self:
            raise KeyError(f'{self} does not have key `{key}`!')
        slices = key_to_method[key]()
    else:
        for key in key_resolve_order:
            if key == SampleBatch.DONES or key in self:
                slices = key_to_method[key]()
                break
        if slices is None:
            raise KeyError(f'{self} does not have keys {key_resolve_order}!')
    assert sum((s.count for s in slices)) == self.count, f'Calling split_by_episode on {self} returns {slices}'
    f'which should in total have {self.count} timesteps!'
    return slices