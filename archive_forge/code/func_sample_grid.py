from abc import ABCMeta, abstractmethod
import warnings
from typing import Any, Dict, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.utils.plotting import SimplePlottingAxes
def sample_grid(self, npts: int, xmin: float=None, xmax: float=None, padding: float=3, width: float=0.1, smearing: str='Gauss') -> 'GridDOSData':
    """Sample the DOS data on an evenly-spaced energy grid

        Args:
            npts: Number of sampled points
            xmin: Minimum sampled x value; if unspecified, a default is chosen
            xmax: Maximum sampled x value; if unspecified, a default is chosen
            padding: If xmin/xmax is unspecified, default value will be padded
                by padding * width to avoid cutting off peaks.
            width: Width of broadening kernel
            smearing: selection of broadening kernel (only 'Gauss' is
                implemented)

        Returns:
            (energy values, sampled DOS)
        """
    if xmin is None:
        xmin = min(self.get_energies()) - padding * width
    if xmax is None:
        xmax = max(self.get_energies()) + padding * width
    energies_grid = np.linspace(xmin, xmax, npts)
    weights_grid = self._sample(energies_grid, width=width, smearing=smearing)
    return GridDOSData(energies_grid, weights_grid, info=self.info.copy())