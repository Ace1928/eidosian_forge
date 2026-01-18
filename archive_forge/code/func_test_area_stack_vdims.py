import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from .test_plot import TestPlotlyPlot
def test_area_stack_vdims(self):
    df = pd.DataFrame({'x': [1, 2, 3], 'y_1': [1, 2, 3], 'y_2': [6, 4, 2], 'y_3': [8, 1, 2]})
    overlay = Overlay([Area(df, kdims='x', vdims=col, label=col) for col in ['y_1', 'y_2', 'y_3']])
    plot = Area.stack(overlay)
    baselines = [np.array([0, 0, 0]), np.array([1.0, 2.0, 3.0]), np.array([7.0, 6.0, 5.0])]
    for n, baseline in zip(plot.data, baselines):
        self.assertEqual(plot.data[n].data.Baseline.to_numpy(), baseline)