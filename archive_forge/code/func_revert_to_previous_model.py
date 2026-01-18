import sys
import logging
import math
import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import PointLight, VBase4, AmbientLight, DirectionalLight, ColorAttrib
from panda3d.core import Geom, GeomVertexFormat, GeomVertexData, GeomVertexWriter
from panda3d.core import GeomTriangles, GeomNode
from direct.task import Task
from models import (
def revert_to_previous_model(self):
    """
        Revert the model index to the previous position in the model sequence array, wrapping around if necessary.
        """
    self.model_index = np.mod(self.model_index - 1, len(self.model_names))
    self.load_specified_model_by_name(self.model_names[self.model_index])