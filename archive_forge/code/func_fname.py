import os
import numpy as np
import threading
from time import time
from .. import config, logging
@property
def fname(self):
    """Get/set the internal filename"""
    return self._fname