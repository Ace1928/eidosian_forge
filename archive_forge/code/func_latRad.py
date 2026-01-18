from plotly.express._core import build_dataframe
from plotly.express._doc import make_docstring
from plotly.express._chart_types import choropleth_mapbox, scatter_mapbox
import numpy as np
import pandas as pd
def latRad(lat):
    sin = np.sin(lat * np.pi / 180)
    radX2 = np.log((1 + sin) / (1 - sin)) / 2
    return max(min(radX2, np.pi), -np.pi) / 2