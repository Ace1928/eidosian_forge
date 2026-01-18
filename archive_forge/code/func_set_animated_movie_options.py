from __future__ import annotations
import itertools
import math
import os
import subprocess
import time
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from pymatgen.core import PeriodicSite, Species, Structure
from pymatgen.util.coord import in_coord_list
def set_animated_movie_options(self, animated_movie_options=None):
    """
        Args:
            animated_movie_options ():
        """
    if animated_movie_options is None:
        self.animated_movie_options = self.DEFAULT_ANIMATED_MOVIE_OPTIONS.copy()
    else:
        self.animated_movie_options = self.DEFAULT_ANIMATED_MOVIE_OPTIONS.copy()
        for key in animated_movie_options:
            if key not in self.DEFAULT_ANIMATED_MOVIE_OPTIONS:
                raise ValueError('Wrong option for animated movie')
        self.animated_movie_options.update(animated_movie_options)