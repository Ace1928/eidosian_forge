import base64
from collections import OrderedDict
import importlib
import io
import zlib
from typing import Any, Dict, Optional, Sequence, Type, Union
import numpy as np
import ray
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.gym import try_import_gymnasium_and_gym
from ray.rllib.utils.error import NotSerializable
from ray.rllib.utils.spaces.flexdict import FlexDict
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.spaces.simplex import Simplex
@DeveloperAPI
def gym_space_to_dict(space: gym.spaces.Space) -> Dict:
    """Serialize a gym Space into a JSON-serializable dict.

    Args:
        space: gym.spaces.Space

    Returns:
        Serialized JSON string.
    """

    def _box(sp: gym.spaces.Box) -> Dict:
        return {'space': 'box', 'low': _serialize_ndarray(sp.low), 'high': _serialize_ndarray(sp.high), 'shape': sp._shape, 'dtype': sp.dtype.str}

    def _discrete(sp: gym.spaces.Discrete) -> Dict:
        d = {'space': 'discrete', 'n': int(sp.n)}
        if hasattr(sp, 'start'):
            d['start'] = int(sp.start)
        return d

    def _multi_binary(sp: gym.spaces.MultiBinary) -> Dict:
        return {'space': 'multi-binary', 'n': sp.n}

    def _multi_discrete(sp: gym.spaces.MultiDiscrete) -> Dict:
        return {'space': 'multi-discrete', 'nvec': _serialize_ndarray(sp.nvec), 'dtype': sp.dtype.str}

    def _tuple(sp: gym.spaces.Tuple) -> Dict:
        return {'space': 'tuple', 'spaces': [gym_space_to_dict(sp) for sp in sp.spaces]}

    def _dict(sp: gym.spaces.Dict) -> Dict:
        return {'space': 'dict', 'spaces': {k: gym_space_to_dict(sp) for k, sp in sp.spaces.items()}}

    def _simplex(sp: Simplex) -> Dict:
        return {'space': 'simplex', 'shape': sp._shape, 'concentration': sp.concentration, 'dtype': sp.dtype.str}

    def _repeated(sp: Repeated) -> Dict:
        return {'space': 'repeated', 'child_space': gym_space_to_dict(sp.child_space), 'max_len': sp.max_len}

    def _flex_dict(sp: FlexDict) -> Dict:
        d = {'space': 'flex_dict'}
        for k, s in sp.spaces:
            d[k] = gym_space_to_dict(s)
        return d

    def _text(sp: 'gym.spaces.Text') -> Dict:
        charset = getattr(sp, 'character_set', None)
        if charset is None:
            charset = getattr(sp, 'charset', None)
        if charset is None:
            raise ValueError('Text space must have a character_set or charset attribute')
        return {'space': 'text', 'min_length': sp.min_length, 'max_length': sp.max_length, 'charset': charset}
    if isinstance(space, gym.spaces.Box):
        return _box(space)
    elif isinstance(space, gym.spaces.Discrete):
        return _discrete(space)
    elif isinstance(space, gym.spaces.MultiBinary):
        return _multi_binary(space)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return _multi_discrete(space)
    elif isinstance(space, gym.spaces.Tuple):
        return _tuple(space)
    elif isinstance(space, gym.spaces.Dict):
        return _dict(space)
    elif isinstance(space, gym.spaces.Text):
        return _text(space)
    elif isinstance(space, Simplex):
        return _simplex(space)
    elif isinstance(space, Repeated):
        return _repeated(space)
    elif isinstance(space, FlexDict):
        return _flex_dict(space)
    elif old_gym and isinstance(space, old_gym.spaces.Box):
        return _box(space)
    elif old_gym and isinstance(space, old_gym.spaces.Discrete):
        return _discrete(space)
    elif old_gym and isinstance(space, old_gym.spaces.MultiDiscrete):
        return _multi_discrete(space)
    elif old_gym and isinstance(space, old_gym.spaces.Tuple):
        return _tuple(space)
    elif old_gym and isinstance(space, old_gym.spaces.Dict):
        return _dict(space)
    elif old_gym and old_gym_text_class and isinstance(space, old_gym_text_class):
        return _text(space)
    else:
        raise ValueError('Unknown space type for serialization, ', type(space))