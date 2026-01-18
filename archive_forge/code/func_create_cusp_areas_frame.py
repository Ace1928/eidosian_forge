import tkinter
import math
import sys
from tkinter import ttk
from .gui_utilities import UniformDictController, FpsLabelUpdater
from .raytracing_view import *
from .hyperboloid_utilities import unit_3_vector_and_distance_to_O13_hyperbolic_translation
from .zoom_slider import Slider, ZoomSlider
def create_cusp_areas_frame(self, parent):
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=0)
    row = 0
    cusp_area_maximum = 1.05 * _maximal_cusp_area(self.widget.manifold)
    for i in range(self.widget.manifold.num_cusps()):
        UniformDictController.create_horizontal_scale(frame, uniform_dict=self.widget.ui_parameter_dict, key='cuspAreas', title='Cusp %d' % i, left_end=0.0, right_end=cusp_area_maximum, row=row, update_function=self.widget.recompute_raytracing_data_and_redraw, index=i)
        row += 1
    frame.rowconfigure(row, weight=1)
    UniformDictController.create_checkbox(frame, self.widget.ui_parameter_dict, 'perspectiveType', update_function=self.checkbox_update, text='Ideal view', row=row, column=1)
    return frame