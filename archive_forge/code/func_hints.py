from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def hints(self, hints_info):
    """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
    if hints_info['csm'] > self.options['csm_max']:
        return []
    return getattr(self, f'{self.hints_type}_hints')(hints_info)