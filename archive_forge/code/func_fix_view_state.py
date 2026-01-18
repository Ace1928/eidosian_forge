from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def fix_view_state(self):
    """
        Fixes view state. Implementation resides with self.raytracing_data,
        e.g., if the view matrix takes the camera outside of the current
        tetrahedron, it would change the view matrix and current tetrahedron
        to fix it.
        """
    self.view_state = self.raytracing_data.update_view_state(self.view_state)