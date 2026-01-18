import logging
import numpy as np
import random
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
@DeveloperAPI
def do_minibatch_sgd(samples, policies, local_worker, num_sgd_iter, sgd_minibatch_size, standardize_fields):
    """Execute minibatch SGD.

    Args:
        samples: Batch of samples to optimize.
        policies: Dictionary of policies to optimize.
        local_worker: Master rollout worker instance.
        num_sgd_iter: Number of epochs of optimization to take.
        sgd_minibatch_size: Size of minibatches to use for optimization.
        standardize_fields: List of sample field names that should be
            normalized prior to optimization.

    Returns:
        averaged info fetches over the last SGD epoch taken.
    """
    samples = samples.as_multi_agent()
    learner_info_builder = LearnerInfoBuilder(num_devices=1)
    for policy_id, policy in policies.items():
        if policy_id not in samples.policy_batches:
            continue
        batch = samples.policy_batches[policy_id]
        for field in standardize_fields:
            batch[field] = standardized(batch[field])
        if policy.is_recurrent() and policy.config['model']['max_seq_len'] > sgd_minibatch_size:
            raise ValueError('`sgd_minibatch_size` ({}) cannot be smaller than`max_seq_len` ({}).'.format(sgd_minibatch_size, policy.config['model']['max_seq_len']))
        for i in range(num_sgd_iter):
            for minibatch in minibatches(batch, sgd_minibatch_size):
                results = local_worker.learn_on_batch(MultiAgentBatch({policy_id: minibatch}, minibatch.count))[policy_id]
                learner_info_builder.add_learn_on_batch_results(results, policy_id)
    learner_info = learner_info_builder.finalize()
    return learner_info