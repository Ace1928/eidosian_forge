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
@lru_cache(maxsize=128)
def add_light(self, light_id: int, light_data: Dict[str, Any]) -> None:
    """
        Adds a new light source to the environment, configuring its properties and effects.
        Utilizes memoization to avoid redundant processing when adding lights with identical configurations.

        Parameters:
            light_id (int): The unique identifier for the light.
            light_data (Dict[str, Any]): A dictionary containing the light's properties such as type, intensity, color, etc.

        Returns:
            None
        """
    structured_data = np.array(list(light_data.values()), dtype=[(key, 'f8') for key in light_data.keys()])
    self.lights[light_id] = structured_data
    logging.debug(f'Light added: {light_id} with data {structured_data}')