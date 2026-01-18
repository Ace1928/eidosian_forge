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
def pull_fillings_from_manifold(self):
    self.filling_dict['fillings'] = self._fillings_from_manifold()
    self.update_filling_sliders()
    self.widget.recompute_raytracing_data_and_redraw()
    self.update_volume_label()
    self.reset_geodesics()