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
def create_navigation_frame(self, parent):
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=0)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=0)
    frame.columnconfigure(3, weight=0)
    row = 0
    UniformDictController.create_horizontal_scale(frame, self.widget.navigation_dict, key='translationVelocity', title='Translation Speed', row=row, left_end=0.1, right_end=1.0)
    self.translate_key_label = ttk.Label(frame, text=_translate_key_labels['QWERTY'])
    self.translate_key_label.grid(row=row, column=3, sticky=tkinter.NSEW)
    row += 1
    UniformDictController.create_horizontal_scale(frame, self.widget.navigation_dict, key='rotationVelocity', title='Rotation Speed', row=row, left_end=0.1, right_end=1.0)
    self.rotate_key_label = ttk.Label(frame, text=_rotate_key_labels['QWERTY'])
    self.rotate_key_label.grid(row=row, column=3, sticky=tkinter.NSEW)
    row += 1
    label = ttk.Label(frame, text=_mouse_gestures_text())
    label.grid(row=row, column=0, columnspan=4)
    return frame