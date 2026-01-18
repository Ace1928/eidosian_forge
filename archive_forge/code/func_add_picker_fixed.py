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
def add_picker_fixed(self):
    """Create a cell picker.Returns:"""
    picker = vtk.vtkCellPicker()

    def annotate_pick(obj, event):
        if picker.GetCellId() < 0 and (not self.show_help):
            self.helptxt_actor.VisibilityOff()
        else:
            mapper = picker.GetMapper()
            if mapper in self.mapper_map:
                output = []
                for site in self.mapper_map[mapper]:
                    row = [f'{site.species_string} - ', ', '.join((f'{c:.3f}' for c in site.frac_coords)), '[' + ', '.join((f'{c:.3f}' for c in site.coords)) + ']']
                    output.append(''.join(row))
                self.helptxt_mapper.SetInput('\n'.join(output))
                self.helptxt_actor.SetPosition(10, 10)
                self.helptxt_actor.VisibilityOn()
                self.show_help = False
    self.picker = picker
    picker.AddObserver('EndPickEvent', annotate_pick)
    self.iren.SetPicker(picker)