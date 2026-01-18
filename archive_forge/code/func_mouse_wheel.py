import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
def mouse_wheel(self, event=None):
    if not self.has_mouse or not self.scrollbar.is_visible:
        return
    low, high = self.scrollbar.get()
    delta = event.delta
    self.canvas.yview_scroll(-delta, 'units')