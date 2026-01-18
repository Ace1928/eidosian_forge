import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def initialize_stationary(self):
    """
        Initialize the statespace model as stationary.
        """
    self.initialize('stationary')