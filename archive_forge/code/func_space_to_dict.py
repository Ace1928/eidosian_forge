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
def space_to_dict(space: gym.spaces.Space) -> Dict:
    d = {'space': gym_space_to_dict(space)}
    if 'original_space' in space.__dict__:
        d['original_space'] = space_to_dict(space.original_space)
    return d