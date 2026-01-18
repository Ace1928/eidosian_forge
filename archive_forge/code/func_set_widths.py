import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
def set_widths(self):
    for n in range(self.num_columns):
        header_width = self.header.grid_bbox(n, 0, n, 0)[2]
        column_width = self.scrollable_frame.grid_bbox(n, 0, n, 0)[2]
        width = max(header_width, column_width, 40)
        self.header.columnconfigure(n, minsize=width)
        self.scrollable_frame.columnconfigure(n, minsize=width)