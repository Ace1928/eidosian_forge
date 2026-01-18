from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def single_cap_hints(self, hints_info):
    """Return hints for an additional neighbors set, i.e. the voronoi indices that
            constitute this new neighbors set, in case of a "Single cap" hint.

            Args:
                hints_info: Info needed to build new "hinted" neighbors set.

            Returns:
                list[int]: Voronoi indices of the new "hinted" neighbors set.
            """
    cap_index_perfect = self.options['cap_index']
    nb_set = hints_info['nb_set']
    permutation = hints_info['permutation']
    nb_set_voronoi_indices_perfect_aligned = nb_set.get_neighb_voronoi_indices(permutation=permutation)
    cap_voronoi_index = nb_set_voronoi_indices_perfect_aligned[cap_index_perfect]
    new_site_voronoi_indices = list(nb_set.site_voronoi_indices)
    new_site_voronoi_indices.remove(cap_voronoi_index)
    return [new_site_voronoi_indices]