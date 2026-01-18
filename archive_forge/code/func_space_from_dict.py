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
def space_from_dict(d: Dict) -> gym.spaces.Space:
    space = gym_space_from_dict(d['space'])
    if 'original_space' in d:
        assert 'space' in d['original_space']
        if isinstance(d['original_space']['space'], str):
            space.original_space = gym_space_from_dict(d['original_space'])
        else:
            space.original_space = space_from_dict(d['original_space'])
    return space