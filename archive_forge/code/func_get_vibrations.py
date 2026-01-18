from math import pi, sqrt, log
import sys
import numpy as np
from pathlib import Path
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils.filecache import get_json_cache
from .data import VibrationsData
from collections import namedtuple
def get_vibrations(self, method='standard', direction='central', read_cache=True, **kw):
    """Get vibrations as VibrationsData object

        If read() has not yet been called, this will be called to assemble data
        from the outputs of run(). Most of the arguments to this function are
        options to be passed to read() in this case.

        Args:
            method (str): Calculation method passed to read()
            direction (str): Finite-difference scheme passed to read()
            read_cache (bool): The VibrationsData object will be cached for
                quick access. Set False to force regeneration of the cache with
                the current atoms/Hessian/indices data.
            **kw: Any remaining keyword arguments are passed to read()

        Returns:
            VibrationsData

        """
    if read_cache and self._vibrations is not None:
        return self._vibrations
    else:
        if self.H is None or method.lower() != self.method or direction.lower() != self.direction:
            self.read(method, direction, **kw)
        return VibrationsData.from_2d(self.atoms, self.H, indices=self.indices)