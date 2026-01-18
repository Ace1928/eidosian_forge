import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def name_generator():
    counter = 0
    while True:
        name = 'Kpt{}'.format(counter)
        yield name
        counter += 1