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
def recompute_hyperbolic_structure(self):
    self.widget.manifold.init_hyperbolic_structure(force_recompute=True)
    self.widget.recompute_raytracing_data_and_redraw()
    self.update_volume_label()
    if self.fillings_changed_callback:
        self.fillings_changed_callback()