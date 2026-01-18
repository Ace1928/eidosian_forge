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
def export_scene(self, filename='vtk_scene', all_data_arrays=False):
    if '.' not in filename:
        filename += '.synch'
    import panel.pane.vtk.synchronizable_serializer as rws
    context = rws.SynchronizationContext(serialize_all_data_arrays=all_data_arrays, debug=self._debug_serializer)
    scene, arrays, annotations = self._serialize_ren_win(self.object, context, binary=True, compression=False)
    with zipfile.ZipFile(filename, mode='w') as zf:
        zf.writestr('index.json', json.dumps(scene))
        for name, data in arrays.items():
            zf.writestr('data/%s' % name, data, zipfile.ZIP_DEFLATED)
        zf.writestr('annotations.json', json.dumps(annotations))
    return filename