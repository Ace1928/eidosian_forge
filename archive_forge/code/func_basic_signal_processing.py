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
def basic_signal_processing(self, signals, simulation_params):
    self._validate_signals(signals)
    return signals * simulation_params.signal_amplification