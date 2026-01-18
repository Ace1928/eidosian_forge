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
def pick_cohomology_class(self, i):
    cohomology_class = self.widget.ui_parameter_dict['cohomology_class'][1]
    for j in range(len(cohomology_class)):
        cohomology_class[j] = 1.0 if i == j else 0.0
    self.widget.recompute_raytracing_data_and_redraw()
    for controller in self.class_controllers:
        controller.update()