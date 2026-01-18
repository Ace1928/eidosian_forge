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
def create_status_frame(self, parent):
    frame = ttk.Frame(parent)
    column = 0
    self.view_scale_label = ttk.Label(frame, text='FOV:')
    self.view_scale_label.grid(row=0, column=column)
    column += 1
    self.view_scale_value_label = ttk.Label(frame)
    self.view_scale_value_label.grid(row=0, column=column)
    column += 1
    self.vol_label = ttk.Label(frame)
    self.vol_label.grid(row=0, column=column)
    column += 1
    self.fps_label = ttk.Label(frame)
    self.fps_label.grid(row=0, column=column)
    return frame