from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
def get_special_points_array(self) -> np.ndarray:
    """Return all special points for this lattice as an array.

        Ordering is consistent with special_point_names."""
    if self._variant.special_points is not None:
        d = self.get_special_points()
        labels = self.special_point_names
        assert len(d) == len(labels)
        points = np.empty((len(d), 3))
        for i, label in enumerate(labels):
            points[i] = d[label]
        return points
    points = self._special_points(variant=self._variant, **self._parameters)
    assert len(points) == len(self.special_point_names)
    return np.array(points)