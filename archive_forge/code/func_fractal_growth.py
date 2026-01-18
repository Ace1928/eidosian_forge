import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
import numpy as np
import matplotlib.pyplot as plt
import io
def fractal_growth(x, t, system_params, fractal_params):
    G_params = (system_params['G_a'], system_params['G_b'], system_params['G_c'], system_params['G_d'])
    return Y_total(t, G_params, system_params['k'], system_params['I'], system_params['lambda'], system_params['n'], system_params['epsilon'], system_params['M_minus'], system_params['threshold']) * B(x, t, (fractal_params['theta'], fractal_params['f'], fractal_params['g']))