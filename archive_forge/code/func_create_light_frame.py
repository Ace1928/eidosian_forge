import tkinter
import math
import sys
import time
from tkinter import ttk
from . import gui_utilities
from .gui_utilities import UniformDictController, FpsLabelUpdater
from .view_scale_controller import ViewScaleController
from .raytracing_view import *
from .geodesics_window import GeodesicsWindow
from .hyperboloid_utilities import unit_3_vector_and_distance_to_O13_hyperbolic_translation
from .zoom_slider import Slider, ZoomSlider
def create_light_frame(self, parent):
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=0)
    row = 0
    if self.has_weights:
        UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='contrast', title='Contrast', row=row, left_end=0.0, right_end=0.25, update_function=self.widget.redraw_if_initialized, format_string='%.3f')
        row += 1
    UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='lightBias', title='Light bias', row=row, left_end=0.3, right_end=4.0, update_function=self.widget.redraw_if_initialized)
    row += 1
    UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='lightFalloff', title='Light falloff', row=row, left_end=0.1, right_end=2.0, update_function=self.widget.redraw_if_initialized)
    row += 1
    UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='brightness', title='Brightness', row=row, left_end=0.3, right_end=3.0, update_function=self.widget.redraw_if_initialized)
    return frame