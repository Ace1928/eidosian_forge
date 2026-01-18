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
def show_failed_dirichlet(self, show):
    if show:
        self.geodesics_status_label.configure(text='Constructing Dirichlet domain failed. Cannot compute length spectrum.', foreground='red')
        self.geodesics_button.configure(state='disabled')
    else:
        self.geodesics_status_label.configure(text='')
        self.geodesics_button.configure(state='normal')