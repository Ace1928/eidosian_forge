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
def test_get_vtkjs_pane_type_from_file():
    file = 'StanfordDragon.vtkjs'
    assert PaneBase.get_pane_type(file) is VTKJS