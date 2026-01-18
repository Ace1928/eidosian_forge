from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
from ray.rllib.core.testing.torch.bc_learner import BCTorchLearner
from ray.rllib.core.testing.tf.bc_module import DiscreteBCTFModule
from ray.rllib.core.testing.tf.bc_learner import BCTfLearner
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ResultDict
@classmethod
def get_default_policy_class(cls, config: AlgorithmConfig):
    if config.framework_str == 'torch':
        return TorchPolicyV2
    elif config.framework_str == 'tf2':
        return EagerTFPolicyV2
    else:
        raise ValueError('Unknown framework: {}'.format(config.framework_str))