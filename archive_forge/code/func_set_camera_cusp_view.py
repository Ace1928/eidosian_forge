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
def set_camera_cusp_view(self, which_cusp):
    self.widget.view_state, view_scale = self.widget.raytracing_data.cusp_view_state_and_scale(which_cusp)
    extra_scale = 1.1
    self.widget.ui_uniform_dict['perspectiveType'][1] = 1
    self.widget.ui_uniform_dict['viewScale'][1] = float(extra_scale * view_scale)
    self.perspective_type_controller.update()
    self.perspective_type_changed()