from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def destroy_widget(self, canvaswidget):
    """
        Remove a canvas widget from this ``CanvasFrame``.  This
        deregisters the canvas widget, and destroys it.
        """
    self.remove_widget(canvaswidget)
    canvaswidget.destroy()