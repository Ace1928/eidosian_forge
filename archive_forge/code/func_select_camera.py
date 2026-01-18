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
def select_camera(self, camera_id: str) -> None:
    """
        Selects a camera as the active camera, directing the rendering process to use its view. Logs the action and handles the case where the camera ID is not found.

        Parameters:
            camera_id (str): The unique identifier for the camera to be activated.
        """
    if camera_id in self.cameras:
        self.active_camera = camera_id
        logging.info(f'Active camera set to: {camera_id}')
    else:
        logging.error(f'Camera ID {camera_id} not found.')
        raise ValueError(f'Camera ID {camera_id} not found.')