import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
@staticmethod
def create_checkbox(container, uniform_dict, key, row, update_function=None, column=0, text='', index=None, component_index=None):
    checkbox = ttk.Checkbutton(container, takefocus=0)
    checkbox.grid(row=row, column=column)
    checkbox.configure(text=text)
    return UniformDictController(uniform_dict, key, checkbox=checkbox, update_function=update_function, index=index, component_index=component_index)