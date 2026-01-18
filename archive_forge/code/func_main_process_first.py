import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
@contextlib.contextmanager
def main_process_first(self, local=True, desc='work'):
    """
        A context manager for torch distributed environment where on needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.

        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.

        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

        """
    if is_torch_available() and self.world_size > 1:
        main_process_desc = 'main local process' if local else 'main process'
        if self.distributed_state is not None:
            is_main_process = self.distributed_state.is_local_main_process if local else self.distributed_state.is_main_process
        elif is_sagemaker_mp_enabled():
            is_main_process = smp.rank() == 0
        try:
            if not is_main_process:
                logger.debug(f'{self.process_index}: waiting for the {main_process_desc} to perform {desc}')
                if is_torch_tpu_available():
                    xm.rendezvous(desc)
                else:
                    dist.barrier()
            yield
        finally:
            if is_main_process:
                logger.debug(f'{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas')
                if is_torch_tpu_available():
                    xm.rendezvous(desc)
                else:
                    dist.barrier()
    else:
        yield