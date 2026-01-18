import contextlib
import dataclasses
import functools
import itertools
import logging
import operator
import re
from collections import namedtuple
from itertools import chain
from typing import (
import sympy
from sympy.printing.printer import Printer
import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from .. import config, metrics
from ..utils import (
from ..virtualized import ops, OpsValue, V
def maybe_append_choice(self, choices, **kwargs):
    """
        Maybe generates a new ChoiceCaller and appends it into existing choices.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """
    try:
        choices.append(self.generate(**kwargs))
    except NotImplementedError:
        pass