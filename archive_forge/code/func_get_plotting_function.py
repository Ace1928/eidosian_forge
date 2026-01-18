import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def get_plotting_function(plot_name, plot_module, backend):
    """Return plotting function for correct backend."""
    _backend = {'mpl': 'matplotlib', 'bokeh': 'bokeh', 'matplotlib': 'matplotlib'}
    if backend is None:
        backend = rcParams['plot.backend']
    backend = backend.lower()
    try:
        backend = _backend[backend]
    except KeyError as err:
        raise KeyError(f'Backend {backend} is not implemented. Try backend in {set(_backend.values())}') from err
    if backend == 'bokeh':
        try:
            import bokeh
            assert packaging.version.parse(bokeh.__version__) >= packaging.version.parse('1.4.0')
        except (ImportError, AssertionError) as err:
            raise ImportError("'bokeh' backend needs Bokeh (1.4.0+) installed. Please upgrade or install") from err
    module = importlib.import_module(f'arviz.plots.backends.{backend}.{plot_module}')
    plotting_method = getattr(module, plot_name)
    return plotting_method