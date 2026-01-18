import numpy as np
def reward_wrapper(self, reward_dict):
    for k in reward_dict.keys():
        reward_dict[k] += np.random.normal(loc=reward_uncertainty_mean, scale=reward_uncertainty_std, size=())
    return reward_dict