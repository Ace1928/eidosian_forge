import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def calculate_properties(self, atoms, properties):
    """This method is experimental; currently for internal use."""
    for name in properties:
        if name not in all_outputs:
            raise ValueError(f'No such property: {name}')
    self.calculate(atoms, properties, system_changes=all_changes)
    props = self.export_properties()
    for name in properties:
        if name not in props:
            raise PropertyNotPresent(name)
    return props