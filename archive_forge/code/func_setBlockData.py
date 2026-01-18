from OpenGL.GL import *  # noqa
from OpenGL.GL import shaders  # noqa
import numpy as np
import re
def setBlockData(self, blockName, data):
    if data is None:
        del self.blockData[blockName]
    else:
        self.blockData[blockName] = data