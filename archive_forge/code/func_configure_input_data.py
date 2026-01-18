import os
from .. import logging
def configure_input_data(obj, data):
    """
    Configure the input data for vtk pipeline object obj.
    Copied from latest version of mayavi
    """
    if vtk_old():
        obj.input = data
    else:
        obj.set_input_data(data)