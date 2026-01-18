import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from matplotlib.animation import FuncAnimation
import logging
import datetime
import sys
import cProfile
def granular_output(self, state, output_params):
    expanded_output = state * output_params['expansion_factor']
    return expanded_output