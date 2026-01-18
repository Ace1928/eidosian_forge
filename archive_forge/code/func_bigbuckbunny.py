import numpy as np
from os.path import dirname
from os.path import join
def bigbuckbunny():
    """Test video data for general video algorithms

    Returns
    -------
    path : string
        Absolute path to bigbuckbunny.mp4
    """
    module_path = dirname(__file__)
    return join(module_path, 'data', 'bigbuckbunny.mp4')