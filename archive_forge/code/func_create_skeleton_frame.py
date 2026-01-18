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
def create_skeleton_frame(self, parent):
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=0)
    row = 0
    UniformDictController.create_horizontal_scale(frame, self.widget.ui_uniform_dict, key='edgeThickness', title='Face boundary thickness', row=row, left_end=0.0, right_end=0.35, update_function=self.widget.redraw_if_initialized, format_string='%.3f')
    row += 1
    self.insphereScaleController = UniformDictController.create_horizontal_scale(frame, self.widget.ui_parameter_dict, key='insphere_scale', title='Insphere scale', row=row, left_end=0.0, right_end=1.25, update_function=self.widget.recompute_raytracing_data_and_redraw, format_string='%.2f')
    row += 1
    self.edgeTubeRadiusController = UniformDictController.create_horizontal_scale(frame, self.widget.ui_parameter_dict, key='edgeTubeRadius', title='Edge thickness', row=row, left_end=0.0, right_end=0.2, update_function=self.widget.redraw_if_initialized)
    row += 1
    label = ttk.Label(frame, text='Edge colors', padding=gui_utilities.label_pad)
    label.grid(row=row, column=0)
    self.edgeColorController = UniformDictController.create_checkbox(frame, self.widget.ui_uniform_dict, key='desaturate_edges', text='desaturate', row=row, column=1, update_function=self.widget.redraw_if_initialized)
    row += 1
    return frame