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
def calculate_pattern_strength(self, input_history, processing_params):
    if len(input_history) == 0:
        return 0
    input_history_array = np.array(input_history)
    fft_result = np.fft.fft(input_history_array, axis=0)
    dominant_frequencies = np.abs(fft_result).mean(axis=0)
    entropy = -np.sum(dominant_frequencies * np.log(dominant_frequencies + 1e-09), axis=0)
    combined_score = np.sum(dominant_frequencies) + entropy
    normalized_score = np.clip(combined_score / processing_params['pattern_params']['max_score'], 0, 1)
    if isinstance(normalized_score, np.ndarray):
        normalized_score = int(np.mean(normalized_score))
    else:
        normalized_score = int(normalized_score)
    return normalized_score