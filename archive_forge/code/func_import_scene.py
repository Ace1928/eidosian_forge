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
@staticmethod
def import_scene(filename, synchronizable=True):
    from .synchronizable_deserializer import import_synch_file
    if synchronizable:
        return VTKRenderWindowSynchronized(import_synch_file(filename=filename), serialize_on_instantiation=False)
    else:
        return VTKRenderWindow(import_synch_file(filename=filename), serialize_on_instantiation=True)