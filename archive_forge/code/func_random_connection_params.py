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
def random_connection_params(simulation_params):
    if not isinstance(simulation_params, SimulationParameters):
        raise TypeError('simulation_params must be an instance of SimulationParameters')
    return {'strength': np.clip(np.random.uniform(0.5, 1.0), 0, 127), 'repeat_factor': np.random.randint(1, 5), 'invert_signal': random.choice([True, False])}