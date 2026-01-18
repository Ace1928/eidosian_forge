import json
import numpy as np
import ase.units as units
from ase import Atoms
from ase.data import chemical_symbols
def nomad_entry_to_images(section):
    """Yield the images from a Nomad entry.

    The entry must contain a section_run.
    One atoms object will be yielded for each section_system."""