from collections import defaultdict, deque
from functools import partial
import pathlib
from typing import (
import uuid
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.learner.learner import LearnerSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.minibatch_utils import ShardBatchIterator
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.train._internal.backend_executor import BackendExecutor
from ray.tune.utils.file_transfer import sync_dir_between_nodes
from ray.util.annotations import PublicAPI
def load_module_state(self, *, marl_module_ckpt_dir: Optional[str]=None, modules_to_load: Optional[Set[str]]=None, rl_module_ckpt_dirs: Optional[Mapping[ModuleID, str]]=None) -> None:
    """Load the checkpoints of the modules being trained by this LearnerGroup.

        `load_module_state` can be used 3 ways:
            1. Load a checkpoint for the MultiAgentRLModule being trained by this
                LearnerGroup. Limit the modules that are loaded from the checkpoint
                by specifying the `modules_to_load` argument.
            2. Load the checkpoint(s) for single agent RLModules that
                are in the MultiAgentRLModule being trained by this LearnerGroup.
            3. Load a checkpoint for the MultiAgentRLModule being trained by this
                LearnerGroup and load the checkpoint(s) for single agent RLModules
                that are in the MultiAgentRLModule. The checkpoints for the single
                agent RLModules take precedence over the module states in the
                MultiAgentRLModule checkpoint.

        NOTE: At lease one of marl_module_ckpt_dir or rl_module_ckpt_dirs is
            must be specified. modules_to_load can only be specified if
            marl_module_ckpt_dir is specified.

        Args:
            marl_module_ckpt_dir: The path to the checkpoint for the
                MultiAgentRLModule.
            modules_to_load: A set of module ids to load from the checkpoint.
            rl_module_ckpt_dirs: A mapping from module ids to the path to a
                checkpoint for a single agent RLModule.
        """
    if not (marl_module_ckpt_dir or rl_module_ckpt_dirs):
        raise ValueError('At least one of multi_agent_module_state or single_agent_module_states must be specified.')
    if marl_module_ckpt_dir:
        if not isinstance(marl_module_ckpt_dir, str):
            raise ValueError('multi_agent_module_state must be a string path.')
        marl_module_ckpt_dir = self._resolve_checkpoint_path(marl_module_ckpt_dir)
    if rl_module_ckpt_dirs:
        if not isinstance(rl_module_ckpt_dirs, dict):
            raise ValueError('single_agent_module_states must be a dictionary.')
        for module_id, path in rl_module_ckpt_dirs.items():
            if not isinstance(path, str):
                raise ValueError('rl_module_ckpt_dirs must be a dictionary mapping module ids to string paths.')
            rl_module_ckpt_dirs[module_id] = self._resolve_checkpoint_path(path)
    if modules_to_load:
        if not isinstance(modules_to_load, set):
            raise ValueError('modules_to_load must be a set.')
        for module_id in modules_to_load:
            if not isinstance(module_id, str):
                raise ValueError('modules_to_load must be a list of strings.')
    if self.is_local:
        module_keys = set(self._learner.module.keys())
    else:
        workers = self._worker_manager.healthy_actor_ids()
        module_keys = set(self._get_results(self._worker_manager.foreach_actor(lambda w: w.module.keys(), remote_actor_ids=[workers[0]]))[0])
    if marl_module_ckpt_dir and rl_module_ckpt_dirs:
        if modules_to_load:
            if any((module_id in modules_to_load for module_id in rl_module_ckpt_dirs.keys())):
                raise ValueError(f'module_id {module_id} was specified in both modules_to_load and rl_module_ckpt_dirs. Please only specify a module to be loaded only once, either in modules_to_load or rl_module_ckpt_dirs, but not both.')
        else:
            modules_to_load = module_keys - set(rl_module_ckpt_dirs.keys())
    if self._is_local:
        if marl_module_ckpt_dir:
            self._learner.module.load_state(marl_module_ckpt_dir, modules_to_load=modules_to_load)
        if rl_module_ckpt_dirs:
            for module_id, path in rl_module_ckpt_dirs.items():
                self._learner.module[module_id].load_state(path / RLMODULE_STATE_DIR_NAME)
    else:
        self._distributed_load_module_state(marl_module_ckpt_dir=marl_module_ckpt_dir, modules_to_load=modules_to_load, rl_module_ckpt_dirs=rl_module_ckpt_dirs)