from OpenGL.GL import *  # noqa
from OpenGL.GL import shaders  # noqa
import numpy as np
import re
def setUniformData(self, uniformName, data):
    if data is None:
        del self.uniformData[uniformName]
    else:
        self.uniformData[uniformName] = data