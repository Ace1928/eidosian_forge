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
def set_active_scene(self, scene_id: str) -> None:
    """
        Sets a loaded scene as the active scene, transitioning the display and interaction focus. Includes error handling to manage non-existent scenes.

        Parameters:
            scene_id (str): The unique identifier for the scene to be activated.
        """
    try:
        if scene_id in self.scenes:
            self.current_scene = scene_id
            logging.info(f'Active scene set to: {scene_id}')
        else:
            raise KeyError(f'Scene ID {scene_id} not found.')
    except KeyError as e:
        logging.error(e)
        print(e)