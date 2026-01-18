from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def moveto(self, x, y, anchor='NW'):
    """
        Move this canvas widget to the given location.  In particular,
        shift the canvas widget such that the corner or side of the
        bounding box specified by ``anchor`` is at location (``x``,
        ``y``).

        :param x,y: The location that the canvas widget should be moved
            to.
        :param anchor: The corner or side of the canvas widget that
            should be moved to the specified location.  ``'N'``
            specifies the top center; ``'NE'`` specifies the top right
            corner; etc.
        """
    x1, y1, x2, y2 = self.bbox()
    if anchor == 'NW':
        self.move(x - x1, y - y1)
    if anchor == 'N':
        self.move(x - x1 / 2 - x2 / 2, y - y1)
    if anchor == 'NE':
        self.move(x - x2, y - y1)
    if anchor == 'E':
        self.move(x - x2, y - y1 / 2 - y2 / 2)
    if anchor == 'SE':
        self.move(x - x2, y - y2)
    if anchor == 'S':
        self.move(x - x1 / 2 - x2 / 2, y - y2)
    if anchor == 'SW':
        self.move(x - x1, y - y2)
    if anchor == 'W':
        self.move(x - x1, y - y1 / 2 - y2 / 2)