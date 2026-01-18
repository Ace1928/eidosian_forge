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
def create_frame_with_main_widget(self, parent, manifold, weights, cohomology_basis, cohomology_class, geodesics):
    frame = ttk.Frame(parent)
    column = 0
    self.widget = RaytracingView('ideal', manifold, weights=weights, cohomology_basis=cohomology_basis, cohomology_class=cohomology_class, geodesics=geodesics, container=frame, width=600, height=500, double=1, depth=1)
    self.widget.grid(row=0, column=column, sticky=tkinter.NSEW)
    self.widget.make_current()
    frame.columnconfigure(column, weight=1)
    frame.rowconfigure(0, weight=1)
    column += 1
    self.view_scale_slider = Slider(frame, left_end=-100.0, right_end=100.0, orient=tkinter.VERTICAL)
    self.view_scale_slider.grid(row=0, column=column, sticky=tkinter.NSEW)
    return frame