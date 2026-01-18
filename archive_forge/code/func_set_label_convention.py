import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Patch3D
from .utils import matplotlib_close_if_inline
def set_label_convention(self, convention):
    """Set x, y and z labels according to one of conventions.

        Args:
            convention (str):
                One of the following:
                    - "original"
                    - "xyz"
                    - "sx sy sz"
                    - "01"
                    - "polarization jones"
                    - "polarization jones letters"
                    see also: http://en.wikipedia.org/wiki/Jones_calculus
                    - "polarization stokes"
                    see also: http://en.wikipedia.org/wiki/Stokes_parameters
        Raises:
            Exception: If convention is not valid.
        """
    ketex = '$\\left.|%s\\right\\rangle$'
    if convention == 'original':
        self.xlabel = ['$x$', '']
        self.ylabel = ['$y$', '']
        self.zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
    elif convention == 'xyz':
        self.xlabel = ['$x$', '']
        self.ylabel = ['$y$', '']
        self.zlabel = ['$z$', '']
    elif convention == 'sx sy sz':
        self.xlabel = ['$s_x$', '']
        self.ylabel = ['$s_y$', '']
        self.zlabel = ['$s_z$', '']
    elif convention == '01':
        self.xlabel = ['', '']
        self.ylabel = ['', '']
        self.zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
    elif convention == 'polarization jones':
        self.xlabel = [ketex % '\\nearrow\\hspace{-1.46}\\swarrow', ketex % '\\nwarrow\\hspace{-1.46}\\searrow']
        self.ylabel = [ketex % '\\circlearrowleft', ketex % '\\circlearrowright']
        self.zlabel = [ketex % '\\leftrightarrow', ketex % '\\updownarrow']
    elif convention == 'polarization jones letters':
        self.xlabel = [ketex % 'D', ketex % 'A']
        self.ylabel = [ketex % 'L', ketex % 'R']
        self.zlabel = [ketex % 'H', ketex % 'V']
    elif convention == 'polarization stokes':
        self.ylabel = ['$\\nearrow\\hspace{-1.46}\\swarrow$', '$\\nwarrow\\hspace{-1.46}\\searrow$']
        self.zlabel = ['$\\circlearrowleft$', '$\\circlearrowright$']
        self.xlabel = ['$\\leftrightarrow$', '$\\updownarrow$']
    else:
        raise Exception('No such convention.')