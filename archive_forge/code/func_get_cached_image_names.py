import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
def get_cached_image_names(self):
    """Get a list of image filenames that have been cached.

        This is useful for debugging and profiling only.

        :rtype: list
        :return: List of str
        """
    self._require_index()
    return list(self._cached_images.keys())