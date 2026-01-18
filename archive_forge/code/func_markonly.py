from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def markonly(self, item):
    """
        Remove any current highlighting, and mark the given item.
        :raise ValueError: If ``item`` is not contained in the list.
        """
    self.unmark()
    self.mark(item)