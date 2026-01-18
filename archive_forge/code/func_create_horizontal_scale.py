import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
@staticmethod
def create_horizontal_scale(container, uniform_dict, key, row, left_end, right_end, update_function=None, column=0, title=None, format_string=None, index=None, component_index=None):
    if title:
        title_label = ttk.Label(container, text=title, padding=label_pad)
        title_label.grid(row=row, column=column, sticky=tkinter.NE)
        column += 1
    scale = Slider(container=container, left_end=left_end, right_end=right_end)
    scale.grid(row=row, column=column, sticky=slider_stick, padx=10)
    column += 1
    value_label = ttk.Label(container, padding=label_pad)
    value_label.grid(row=row, column=column, sticky=tkinter.NW, padx=20)
    if title:
        title_label.grid_configure(sticky=tkinter.N, pady=4)
    return UniformDictController(uniform_dict, key, scale=scale, label=value_label, update_function=update_function, format_string=format_string, index=index, component_index=component_index)