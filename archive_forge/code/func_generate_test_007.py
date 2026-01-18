import os
import xarray as xr
import datashader as ds
import pandas as pd
import numpy as np
import pytest
def generate_test_007():
    points = [((0.5, 0.5), (1.5, 48.5)), ((0.5, 0.5), (48.5, 1.5)), ((48.5, 48.5), (47.5, 0.5)), ((48.5, 48.5), (0.5, 47.5))]
    return (points, 'test_007')