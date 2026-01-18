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
def add_partial_sphere(self, coords, radius, color, start=0, end=360, opacity=1.0):
    """
        Adding a partial sphere (to display partial occupancies.

        Args:
            coords (nd.array): Coordinates
            radius (float): Radius of sphere
            color (): Color of sphere.
            start (float): Starting angle.
            end (float): Ending angle.
            opacity (float): Opacity.
        """
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(coords)
    sphere.SetRadius(radius)
    sphere.SetThetaResolution(18)
    sphere.SetPhiResolution(18)
    sphere.SetStartTheta(start)
    sphere.SetEndTheta(end)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)
    self.ren.AddActor(actor)
    return mapper