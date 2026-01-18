import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def initialize_diffuse(self):
    """
        Initialize the statespace model as diffuse.
        """
    self.initialize('diffuse')