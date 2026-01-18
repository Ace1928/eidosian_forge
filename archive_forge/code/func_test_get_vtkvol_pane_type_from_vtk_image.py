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
def test_get_vtkvol_pane_type_from_vtk_image():
    image_data = make_image_data()
    assert PaneBase.get_pane_type(image_data) is VTKVolume