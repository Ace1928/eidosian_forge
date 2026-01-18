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
def compile_shader(self, source_code: str, shader_type: str) -> None:
    """
        Compiles a shader from source code and logs the operation.

        Parameters:
        source_code (str): The GLSL source code of the shader.
        shader_type (str): The type of the shader (e.g., 'vertex', 'fragment').

        Returns:
        None
        """
    shader_id = self._generate_shader_id(source_code, shader_type)
    self.shaders[shader_id] = (source_code, shader_type)
    logging.debug(f'Shader compiled and stored: ID={shader_id}, Type={shader_type}')