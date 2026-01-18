from __future__ import annotations
import base64
import json
import sys
import zipfile
from abc import abstractmethod
from typing import (
from urllib.request import urlopen
import numpy as np
import param
from bokeh.models import LinearColorMapper
from bokeh.util.serialization import make_globally_unique_id
from pyviz_comms import JupyterComm
from ...param import ParamMethod
from ...util import isfile, lazy_load
from ..base import PaneBase
from ..plot import Bokeh
from .enums import PRESET_CMAPS
def link_camera(self, other):
    """
        Associate the camera of an other VTKSynchronized pane to this renderer
        """
    if not isinstance(other, VTKRenderWindowSynchronized):
        raise TypeError('Only instance of VTKRenderWindow class can be linked')
    else:
        self.vtk_camera = other.vtk_camera