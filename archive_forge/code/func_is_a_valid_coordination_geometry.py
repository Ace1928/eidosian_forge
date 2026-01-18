from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
def is_a_valid_coordination_geometry(self, mp_symbol=None, IUPAC_symbol=None, IUCr_symbol=None, name=None, cn=None) -> bool:
    """
        Checks whether a given coordination geometry is valid (exists) and whether the parameters are coherent with
        each other.

        Args:
            mp_symbol: The mp_symbol of the coordination geometry.
            IUPAC_symbol: The IUPAC_symbol of the coordination geometry.
            IUCr_symbol: The IUCr_symbol of the coordination geometry.
            name: The name of the coordination geometry.
            cn: The coordination of the coordination geometry.
        """
    if name is not None:
        raise NotImplementedError('is_a_valid_coordination_geometry not implemented for the name')
    if mp_symbol is None and IUPAC_symbol is None and (IUCr_symbol is None):
        raise SyntaxError('missing argument for is_a_valid_coordination_geometry : at least one of mp_symbol, IUPAC_symbol and IUCr_symbol must be passed to the function')
    if mp_symbol is not None:
        try:
            cg = self.get_geometry_from_mp_symbol(mp_symbol)
            if IUPAC_symbol is not None and IUPAC_symbol != cg.IUPAC_symbol:
                return False
            if IUCr_symbol is not None and IUCr_symbol != cg.IUCr_symbol:
                return False
            if cn is not None and int(cn) != int(cg.coordination_number):
                return False
            return True
        except LookupError:
            return False
    elif IUPAC_symbol is not None:
        try:
            cg = self.get_geometry_from_IUPAC_symbol(IUPAC_symbol)
            if IUCr_symbol is not None and IUCr_symbol != cg.IUCr_symbol:
                return False
            if cn is not None and cn != cg.coordination_number:
                return False
            return True
        except LookupError:
            return False
    elif IUCr_symbol is not None:
        try:
            cg = self.get_geometry_from_IUCr_symbol(IUCr_symbol)
            if cn is not None and cn != cg.coordination_number:
                return False
            return True
        except LookupError:
            return True
    raise RuntimeError('Should not be here!')