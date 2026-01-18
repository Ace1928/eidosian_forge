import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import (
from torch.overrides import (
from torch.utils.dlpack import DLDeviceType
def reinforce(self, reward):

    def trim(str):
        return '\n'.join([line.strip() for line in str.split('\n')])
    raise RuntimeError(trim('reinforce() was removed.\n            Use torch.distributions instead.\n            See https://pytorch.org/docs/master/distributions.html\n\n            Instead of:\n\n            probs = policy_network(state)\n            action = probs.multinomial()\n            next_state, reward = env.step(action)\n            action.reinforce(reward)\n            action.backward()\n\n            Use:\n\n            probs = policy_network(state)\n            # NOTE: categorical is equivalent to what used to be called multinomial\n            m = torch.distributions.Categorical(probs)\n            action = m.sample()\n            next_state, reward = env.step(action)\n            loss = -m.log_prob(action) * reward\n            loss.backward()\n        '))