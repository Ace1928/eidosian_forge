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
def create_cusps_frame(self, parent):
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=0)
    row = 0
    cusp_area_maximum = 1.05 * _maximal_cusp_area(self.widget.manifold)
    for i in range(self.widget.manifold.num_cusps()):
        UniformDictController.create_horizontal_scale(frame, uniform_dict=self.widget.ui_parameter_dict, key='cuspAreas', title='Cusp %d' % i, left_end=0.0, right_end=cusp_area_maximum, row=row, update_function=self.widget.recompute_raytracing_data_and_redraw, index=i)
        cusp_button = ttk.Button(frame, text='View', takefocus=0, command=lambda which_cusp=i: self.set_camera_cusp_view(which_cusp))
        cusp_button.grid(row=row, column=3)
        row += 1
    frame.rowconfigure(row, weight=1)
    view_frame = ttk.Frame(frame)
    view_frame.grid(row=row, column=1)
    view_label = ttk.Label(view_frame, text='View:')
    view_label.grid(row=0, column=0)
    radio_buttons = []
    for i, text in enumerate(['Material', 'Ideal', 'Hyperideal']):
        button = ttk.Radiobutton(view_frame, value=i, text=text, takefocus=0)
        button.grid(row=0, column=i + 1)
        radio_buttons.append(button)
    self.perspective_type_controller = UniformDictController(self.widget.ui_uniform_dict, key='perspectiveType', radio_buttons=radio_buttons, update_function=self.perspective_type_changed)
    self.geodesics_button = ttk.Button(frame, text='Geodesics ...', takefocus=0, command=self.show_geodesics_window)
    self.geodesics_button.grid(row=row, column=3)
    row += 1
    self.geodesics_status_label = ttk.Label(frame, text='')
    self.geodesics_status_label.grid(row=row, column=0, columnspan=4)
    return frame