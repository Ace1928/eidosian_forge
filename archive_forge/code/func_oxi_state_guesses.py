from __future__ import annotations
import collections
import os
import re
import string
import warnings
from functools import total_ordering
from itertools import combinations_with_replacement, product
from math import isnan
from typing import TYPE_CHECKING, cast
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.periodic_table import DummySpecies, Element, ElementType, Species, get_el_sp
from pymatgen.core.units import Mass
from pymatgen.util.string import Stringify, formula_double_format
def oxi_state_guesses(self, oxi_states_override: dict | None=None, target_charge: float=0, all_oxi_states: bool=False, max_sites: int | None=None) -> tuple[dict[str, float]]:
    """Checks if the composition is charge-balanced and returns back all
        charge-balanced oxidation state combinations. Composition must have
        integer values. Note that more num_atoms in the composition gives
        more degrees of freedom. e.g., if possible oxidation states of
        element X are [2,4] and Y are [-3], then XY is not charge balanced
        but X2Y2 is. Results are returned from most to least probable based
        on ICSD statistics. Use max_sites to improve performance if needed.

        Args:
            oxi_states_override (dict): dict of str->list to override an
                element's common oxidation states, e.g. {"V": [2,3,4,5]}
            target_charge (int): the desired total charge on the structure.
                Default is 0 signifying charge balance.
            all_oxi_states (bool): if True, an element defaults to
                all oxidation states in pymatgen Element.icsd_oxidation_states.
                Otherwise, default is Element.common_oxidation_states. Note
                that the full oxidation state list is *very* inclusive and
                can produce nonsensical results.
            max_sites (int): if possible, will reduce Compositions to at most
                this many sites to speed up oxidation state guesses. If the
                composition cannot be reduced to this many sites a ValueError
                will be raised. Set to -1 to just reduce fully. If set to a
                number less than -1, the formula will be fully reduced but a
                ValueError will be thrown if the number of atoms in the reduced
                formula is greater than abs(max_sites).

        Returns:
            list[dict]: each dict reports an element symbol and average
                oxidation state across all sites in that composition. If the
                composition is not charge balanced, an empty list is returned.
        """
    if len(self.elements) == 1:
        return ({self.elements[0].symbol: 0.0},)
    return self._get_oxi_state_guesses(all_oxi_states, max_sites, oxi_states_override, target_charge)[0]