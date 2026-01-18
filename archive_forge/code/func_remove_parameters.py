from __future__ import annotations
import copy
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.inputs import AimsControlIn, AimsGeometryIn
from pymatgen.io.aims.parsers import AimsParseError, read_aims_output
from pymatgen.io.core import InputFile, InputGenerator, InputSet
def remove_parameters(self, keys: Iterable[str] | str, strict: bool=True) -> dict[str, Any]:
    """Remove the aims parameters listed in keys.

        This removes the aims variables from the parameters object.

        Args:
            keys (Iterable[str] or str): string or list of strings with the names of
                the aims parameters to be removed.
            strict (bool): whether to raise a KeyError if one of the aims parameters
                to be removed is not present.

        Returns:
            dict[str, Any]: Dictionary with the variables that have been removed.
        """
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key not in self._parameters:
            if strict:
                raise ValueError(f'key={key!r} not in list(self._parameters)={list(self._parameters)!r}')
            continue
        del self._parameters[key]
    return self.set_parameters(**self._parameters)