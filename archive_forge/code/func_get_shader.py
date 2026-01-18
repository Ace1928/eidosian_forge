import pyopencl as cl  # https://documen.tician.de/pyopencl/ - Used for managing and executing OpenCL commands on GPUs.
import OpenGL.GL as gl  # https://pyopengl.sourceforge.io/documentation/ - Used for executing OpenGL commands for rendering graphics.
import json  # https://docs.python.org/3/library/json.html - Used for parsing and outputting JSON formatted data.
import numpy as np  # https://numpy.org/doc/ - Used for numerical operations on arrays and matrices.
import functools  # https://docs.python.org/3/library/functools.html - Provides higher-order functions and operations on callable objects.
import logging  # https://docs.python.org/3/library/logging.html - Used for logging events and messages during execution.
from pyopencl import (
import hashlib  # https://docs.python.org/3/library/hashlib.html - Used for hashing algorithms.
import pickle  # https://docs.python.org/3/library/pickle.html - Used for serializing and deserializing Python objects.
from typing import (
from functools import (
def get_shader(self, shader_id: int) -> Optional[Tuple[str, str]]:
    """
        Retrieves a compiled shader by its identifier.

        Parameters:
        shader_id (int): The unique identifier of the shader.

        Returns:
        Optional[Tuple[str, str]]: The shader source code and type if found, None otherwise.
        """
    shader = self.shaders.get(shader_id, None)
    if shader is None:
        logging.error(f'Shader ID {shader_id} not found.')
    else:
        logging.debug(f'Shader retrieved: ID={shader_id}, Type={shader[1]}')
    return shader