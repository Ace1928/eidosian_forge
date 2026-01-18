import itertools
import threading
import time
import numpy as np
from numpy.testing import assert_equal
import pytest
import scipy.interpolate
def calibrate_delay(requested_time):
    for n_points in itertools.count(5000, 1000):
        args = generate_params(n_points)
        time_started = time.time()
        interpolate(*args)
        if time.time() - time_started > requested_time:
            return args