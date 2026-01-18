import sys
import os
@staticmethod
def frame_file_is_setup(frame):
    """
        Return True if the indicated frame suggests a setup.py file.
        """
    return frame.f_globals.get('__file__', '').endswith('setup.py')