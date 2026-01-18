import math
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch
def get_len(b):
    return len(b)