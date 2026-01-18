import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def continuous_action_config(action_min_gaussian_std: float=0.001, action_gaussian_std_fn: str='softplus', action_std_for_zero_param: float=1, action_postprocessor: str='Tanh') -> ContinuousDistributionConfig:
    """Configures continuous distributions from numerical and string inputs.

  Currently, only NormalSquashedDistribution is supported. The default
  configuration corresponds to a normal distribution with standard deviation
  computed from params using an unshifted softplus, followed by tanh.
  Args:
    action_min_gaussian_std: minimal standard deviation.
    action_gaussian_std_fn: transform for standard deviation parameters.
    action_std_for_zero_param: shifts the transform to get this std when
      parameters are zero.
    action_postprocessor: the non-linearity applied to the sample from the
      gaussian.

  Returns:
    A continuous distribution setup, with the parameters transform
    to get the standard deviation applied with a shift, as configured.
  """
    config = ContinuousDistributionConfig()
    config.min_gaussian_std = float(action_min_gaussian_std)
    if action_gaussian_std_fn == 'safe_exp':
        config.gaussian_std_fn = safe_exp_std_fn(action_std_for_zero_param, config.min_gaussian_std)
    elif action_gaussian_std_fn == 'softplus':
        config.gaussian_std_fn = softplus_std_fn(action_std_for_zero_param, config.min_gaussian_std)
    else:
        raise ValueError(f'Flag `action_gaussian_std_fn` only supports safe_exp and softplus, got: {action_gaussian_std_fn}')
    config.postprocessor = action_postprocessor
    return config