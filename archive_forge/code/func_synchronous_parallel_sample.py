import logging
from typing import List, Optional, Union
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import (
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.typing import SampleBatchType
@ExperimentalAPI
def synchronous_parallel_sample(*, worker_set: WorkerSet, max_agent_steps: Optional[int]=None, max_env_steps: Optional[int]=None, concat: bool=True) -> Union[List[SampleBatchType], SampleBatchType]:
    """Runs parallel and synchronous rollouts on all remote workers.

    Waits for all workers to return from the remote calls.

    If no remote workers exist (num_workers == 0), use the local worker
    for sampling.

    Alternatively to calling `worker.sample.remote()`, the user can provide a
    `remote_fn()`, which will be applied to the worker(s) instead.

    Args:
        worker_set: The WorkerSet to use for sampling.
        remote_fn: If provided, use `worker.apply.remote(remote_fn)` instead
            of `worker.sample.remote()` to generate the requests.
        max_agent_steps: Optional number of agent steps to be included in the
            final batch.
        max_env_steps: Optional number of environment steps to be included in the
            final batch.
        concat: Whether to concat all resulting batches at the end and return the
            concat'd batch.

    Returns:
        The list of collected sample batch types (one for each parallel
        rollout worker in the given `worker_set`).

    .. testcode::

        # Define an RLlib Algorithm.
        from ray.rllib.algorithms.ppo import PPO, PPOConfig
        config = PPOConfig().environment("CartPole-v1")
        algorithm = PPO(config=config)
        # 2 remote workers (num_workers=2):
        batches = synchronous_parallel_sample(worker_set=algorithm.workers,
            concat=False)
        print(len(batches))

    .. testoutput::

        2
    """
    assert not (max_agent_steps is not None and max_env_steps is not None)
    agent_or_env_steps = 0
    max_agent_or_env_steps = max_agent_steps or max_env_steps or None
    all_sample_batches = []
    while max_agent_or_env_steps is None and agent_or_env_steps == 0 or (max_agent_or_env_steps is not None and agent_or_env_steps < max_agent_or_env_steps):
        if worker_set.num_remote_workers() <= 0:
            sample_batches = [worker_set.local_worker().sample()]
        else:
            sample_batches = worker_set.foreach_worker(lambda w: w.sample(), local_worker=False, healthy_only=True)
            if worker_set.num_healthy_remote_workers() <= 0:
                break
        for b in sample_batches:
            if max_agent_steps:
                agent_or_env_steps += b.agent_steps()
            else:
                agent_or_env_steps += b.env_steps()
        all_sample_batches.extend(sample_batches)
    if concat is True:
        full_batch = concat_samples(all_sample_batches)
        return full_batch
    else:
        return all_sample_batches