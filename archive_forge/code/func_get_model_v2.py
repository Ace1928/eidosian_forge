from functools import partial
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional, Type, Union
from ray.tune.registry import (
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@staticmethod
@DeveloperAPI
def get_model_v2(obs_space: gym.Space, action_space: gym.Space, num_outputs: int, model_config: ModelConfigDict, framework: str='tf', name: str='default_model', model_interface: type=None, default_model: type=None, **model_kwargs) -> ModelV2:
    """Returns a suitable model compatible with given spaces and output.

        Args:
            obs_space: Observation space of the target gym env. This
                may have an `original_space` attribute that specifies how to
                unflatten the tensor into a ragged tensor.
            action_space: Action space of the target gym env.
            num_outputs: The size of the output vector of the model.
            model_config: The "model" sub-config dict
                within the Algorithm's config dict.
            framework: One of "tf2", "tf", "torch", or "jax".
            name: Name (scope) for the model.
            model_interface: Interface required for the model
            default_model: Override the default class for the model. This
                only has an effect when not using a custom model
            model_kwargs: Args to pass to the ModelV2 constructor

        Returns:
            model (ModelV2): Model to use for the policy.
        """
    ModelCatalog._validate_config(config=model_config, action_space=action_space, framework=framework)
    if model_config.get('custom_model'):
        customized_model_kwargs = dict(model_kwargs, **model_config.get('custom_model_config', {}))
        if isinstance(model_config['custom_model'], type):
            model_cls = model_config['custom_model']
        elif isinstance(model_config['custom_model'], str) and '.' in model_config['custom_model']:
            return from_config(cls=model_config['custom_model'], obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=customized_model_kwargs, name=name)
        else:
            model_cls = _global_registry.get(RLLIB_MODEL, model_config['custom_model'])
        if not issubclass(model_cls, ModelV2):
            if framework not in ['tf', 'tf2'] or not issubclass(model_cls, tf.keras.Model):
                raise ValueError('`model_cls` must be a ModelV2 sub-class, but is {}!'.format(model_cls))
        logger.info('Wrapping {} as {}'.format(model_cls, model_interface))
        model_cls = ModelCatalog._wrap_if_needed(model_cls, model_interface)
        if framework in ['tf2', 'tf']:
            if model_config.get('use_lstm') or model_config.get('use_attention'):
                from ray.rllib.models.tf.attention_net import AttentionWrapper
                from ray.rllib.models.tf.recurrent_net import LSTMWrapper
                wrapped_cls = model_cls
                forward = wrapped_cls.forward
                model_cls = ModelCatalog._wrap_if_needed(wrapped_cls, LSTMWrapper if model_config.get('use_lstm') else AttentionWrapper)
                model_cls._wrapped_forward = forward
            created = set()

            def track_var_creation(next_creator, **kw):
                v = next_creator(**kw)
                created.add(v.ref())
                return v
            with tf.variable_creator_scope(track_var_creation):
                if issubclass(model_cls, tf.keras.Model):
                    instance = model_cls(input_space=obs_space, action_space=action_space, num_outputs=num_outputs, name=name, **customized_model_kwargs)
                else:
                    try:
                        instance = model_cls(obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs)
                    except TypeError as e:
                        if '__init__() got an unexpected ' in e.args[0]:
                            instance = model_cls(obs_space, action_space, num_outputs, model_config, name, **model_kwargs)
                            logger.warning("Custom ModelV2 should accept all custom options as **kwargs, instead of expecting them in config['custom_model_config']!")
                        else:
                            raise e
            registered = []
            if not isinstance(instance, tf.keras.Model):
                registered = set(instance.var_list)
            if len(registered) > 0:
                not_registered = set()
                for var in created:
                    if var not in registered:
                        not_registered.add(var)
                if not_registered:
                    raise ValueError("It looks like you are still using `{}.register_variables()` to register your model's weights. This is no longer required, but if you are still calling this method at least once, you must make sure to register all created variables properly. The missing variables are {}, and you only registered {}. Did you forget to call `register_variables()` on some of the variables in question?".format(instance, not_registered, registered))
        elif framework == 'torch':
            if model_config.get('use_lstm') or model_config.get('use_attention'):
                from ray.rllib.models.torch.attention_net import AttentionWrapper
                from ray.rllib.models.torch.recurrent_net import LSTMWrapper
                wrapped_cls = model_cls
                forward = wrapped_cls.forward
                model_cls = ModelCatalog._wrap_if_needed(wrapped_cls, LSTMWrapper if model_config.get('use_lstm') else AttentionWrapper)
                model_cls._wrapped_forward = forward
            try:
                instance = model_cls(obs_space, action_space, num_outputs, model_config, name, **customized_model_kwargs)
            except TypeError as e:
                if '__init__() got an unexpected ' in e.args[0]:
                    instance = model_cls(obs_space, action_space, num_outputs, model_config, name, **model_kwargs)
                    logger.warning("Custom ModelV2 should accept all custom options as **kwargs, instead of expecting them in config['custom_model_config']!")
                else:
                    raise e
        else:
            raise NotImplementedError("`framework` must be 'tf2|tf|torch', but is {}!".format(framework))
        return instance
    if framework in ['tf', 'tf2']:
        v2_class = None
        if not model_config.get('custom_model'):
            v2_class = default_model or ModelCatalog._get_v2_model_class(obs_space, model_config, framework=framework)
        if not v2_class:
            raise ValueError('ModelV2 class could not be determined!')
        if model_config.get('use_lstm') or model_config.get('use_attention'):
            from ray.rllib.models.tf.attention_net import AttentionWrapper
            from ray.rllib.models.tf.recurrent_net import LSTMWrapper
            wrapped_cls = v2_class
            if model_config.get('use_lstm'):
                v2_class = ModelCatalog._wrap_if_needed(wrapped_cls, LSTMWrapper)
                v2_class._wrapped_forward = wrapped_cls.forward
            else:
                v2_class = ModelCatalog._wrap_if_needed(wrapped_cls, AttentionWrapper)
                v2_class._wrapped_forward = wrapped_cls.forward
        wrapper = ModelCatalog._wrap_if_needed(v2_class, model_interface)
        if issubclass(wrapper, tf.keras.Model):
            model = wrapper(input_space=obs_space, action_space=action_space, num_outputs=num_outputs, name=name, **dict(model_kwargs, **model_config))
            return model
        return wrapper(obs_space, action_space, num_outputs, model_config, name, **model_kwargs)
    elif framework == 'torch':
        if not model_config.get('custom_model'):
            v2_class = default_model or ModelCatalog._get_v2_model_class(obs_space, model_config, framework=framework)
        if not v2_class:
            raise ValueError('ModelV2 class could not be determined!')
        if model_config.get('use_lstm') or model_config.get('use_attention'):
            from ray.rllib.models.torch.attention_net import AttentionWrapper
            from ray.rllib.models.torch.recurrent_net import LSTMWrapper
            wrapped_cls = v2_class
            forward = wrapped_cls.forward
            if model_config.get('use_lstm'):
                v2_class = ModelCatalog._wrap_if_needed(wrapped_cls, LSTMWrapper)
            else:
                v2_class = ModelCatalog._wrap_if_needed(wrapped_cls, AttentionWrapper)
            v2_class._wrapped_forward = forward
        wrapper = ModelCatalog._wrap_if_needed(v2_class, model_interface)
        return wrapper(obs_space, action_space, num_outputs, model_config, name, **model_kwargs)
    elif framework == 'jax':
        v2_class = default_model or ModelCatalog._get_v2_model_class(obs_space, model_config, framework=framework)
        wrapper = ModelCatalog._wrap_if_needed(v2_class, model_interface)
        return wrapper(obs_space, action_space, num_outputs, model_config, name, **model_kwargs)
    else:
        raise NotImplementedError("`framework` must be 'tf2|tf|torch', but is {}!".format(framework))