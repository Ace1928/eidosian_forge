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
def create_cohomology_class_frame(self, parent):
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=0)
    row = 0
    self.class_controllers = []
    n = len(self.widget.ui_parameter_dict['cohomology_class'][1])
    for i in range(n):
        button = ttk.Button(frame, text='Class %d' % i, takefocus=0, command=lambda i=i: self.pick_cohomology_class(i))
        button.grid(row=row, column=0)
        self.class_controllers.append(UniformDictController.create_horizontal_scale(frame, column=1, uniform_dict=self.widget.ui_parameter_dict, key='cohomology_class', left_end=-1.0, right_end=1.0, row=row, update_function=self.widget.recompute_raytracing_data_and_redraw, index=i))
        row += 1
    frame.rowconfigure(row, weight=1)
    UniformDictController.create_checkbox(frame, self.widget.ui_uniform_dict, 'showElevation', update_function=self.checkbox_update, text='Elevation', row=row, column=1)
    return frame