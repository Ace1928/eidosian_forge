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
def apply_settings(self, settings):
    keyboard = settings.get('keyboard', 'QWERTY')
    self.translate_key_label.configure(text=_translate_key_labels[keyboard])
    self.rotate_key_label.configure(text=_rotate_key_labels[keyboard])
    self.widget.apply_settings(settings)