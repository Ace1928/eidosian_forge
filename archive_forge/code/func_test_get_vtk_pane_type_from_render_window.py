import base64
import os
import sys
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pytest
from bokeh.models import ColorBar
from panel.models.vtk import (
from panel.pane import VTK, PaneBase, VTKVolume
from panel.pane.vtk.vtk import (
@vtk_available
def test_get_vtk_pane_type_from_render_window():
    assert PaneBase.get_pane_type(vtk.vtkRenderWindow()) is VTKRenderWindowSynchronized
    assert PaneBase.get_pane_type(vtk.vtkRenderWindow(), serialize_on_instantiation=True) is VTKRenderWindow