import numpy as np
from os.path import dirname
from os.path import join
def bikes():
    """Test video data for video scene detection algorithms

    Returns
    -------
    path : string
        Absolute path to bikes.mp4
    """
    module_path = dirname(__file__)
    return join(module_path, 'data', 'bikes.mp4')