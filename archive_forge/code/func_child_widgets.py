from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def child_widgets(self):
    """
        :return: A list of the hierarchical children of this canvas
            widget.  These children are considered part of ``self``
            for purposes of user interaction.
        :rtype: list of CanvasWidget
        """
    return self.__children