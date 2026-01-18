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
def gym_space_from_dict(d: Dict) -> gym.spaces.Space:
    """De-serialize a dict into gym Space.

    Args:
        str: serialized JSON str.

    Returns:
        De-serialized gym space.
    """

    def __common(d: Dict):
        """Common updates to the dict before we use it to construct spaces"""
        ret = d.copy()
        del ret['space']
        if 'dtype' in ret:
            ret['dtype'] = np.dtype(ret['dtype'])
        return ret

    def _box(d: Dict) -> gym.spaces.Box:
        ret = d.copy()
        ret.update({'low': _deserialize_ndarray(d['low']), 'high': _deserialize_ndarray(d['high'])})
        return gym.spaces.Box(**__common(ret))

    def _discrete(d: Dict) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(**__common(d))

    def _multi_binary(d: Dict) -> gym.spaces.MultiBinary:
        return gym.spaces.MultiBinary(**__common(d))

    def _multi_discrete(d: Dict) -> gym.spaces.MultiDiscrete:
        ret = d.copy()
        ret.update({'nvec': _deserialize_ndarray(ret['nvec'])})
        return gym.spaces.MultiDiscrete(**__common(ret))

    def _tuple(d: Dict) -> gym.spaces.Discrete:
        spaces = [gym_space_from_dict(sp) for sp in d['spaces']]
        return gym.spaces.Tuple(spaces=spaces)

    def _dict(d: Dict) -> gym.spaces.Discrete:
        spaces = OrderedDict({k: gym_space_from_dict(sp) for k, sp in d['spaces'].items()})
        return gym.spaces.Dict(spaces=spaces)

    def _simplex(d: Dict) -> Simplex:
        return Simplex(**__common(d))

    def _repeated(d: Dict) -> Repeated:
        child_space = gym_space_from_dict(d['child_space'])
        return Repeated(child_space=child_space, max_len=d['max_len'])

    def _flex_dict(d: Dict) -> FlexDict:
        spaces = {k: gym_space_from_dict(s) for k, s in d.items() if k != 'space'}
        return FlexDict(spaces=spaces)

    def _text(d: Dict) -> 'gym.spaces.Text':
        return gym.spaces.Text(**__common(d))
    space_map = {'box': _box, 'discrete': _discrete, 'multi-binary': _multi_binary, 'multi-discrete': _multi_discrete, 'tuple': _tuple, 'dict': _dict, 'simplex': _simplex, 'repeated': _repeated, 'flex_dict': _flex_dict, 'text': _text}
    space_type = d['space']
    if space_type not in space_map:
        raise ValueError('Unknown space type for de-serialization, ', space_type)
    return space_map[space_type](d)