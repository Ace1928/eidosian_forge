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
def make_orbifold(self):
    for f in self.filling_dict['fillings'][1]:
        for i in [0, 1]:
            f[i] = float(round(f[i]))
    self.update_filling_sliders()
    self.push_fillings_to_manifold()