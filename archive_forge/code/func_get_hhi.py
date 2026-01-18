from __future__ import annotations
import os
from monty.design_patterns import singleton
from pymatgen.core import Composition, Element
def get_hhi(self, comp_or_form):
    """
        Gets the reserve and production HHI for a compound.

        Args:
            comp_or_form (Composition or String): A Composition or String formula

        Returns:
            A tuple representing the (HHI_production, HHI_reserve)
        """
    try:
        if not isinstance(comp_or_form, Composition):
            comp_or_form = Composition(comp_or_form)
        hhi_p = 0
        hhi_r = 0
        for e in comp_or_form.elements:
            percent = comp_or_form.get_wt_fraction(e)
            dp, dr = self._get_hhi_el(e)
            hhi_p += dp * percent
            hhi_r += dr * percent
        return (hhi_p, hhi_r)
    except Exception:
        return (None, None)