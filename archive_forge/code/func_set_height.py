from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def set_height(self, height):
    """
        Change the height of this space widget.

        :param height: The new height.
        :type height: int
        :rtype: None
        """
    [x1, y1, x2, y2] = self.bbox()
    self.canvas().coords(self._tag, x1, y1, x2, y1 + height)