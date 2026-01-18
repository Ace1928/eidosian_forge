import holoviews as hv
from ..util import with_hv_extension, is_polars
from .core import hvPlot, hvPlotTabular   # noqa
from .andrews_curves import andrews_curves   # noqa
from .parallel_coordinates import parallel_coordinates   # noqa
from .lag_plot import lag_plot   # noqa
from .scatter_matrix import scatter_matrix   # noqa
def boxplot_series(*args, **kwargs):
    return plot(*args, kind='box', **kwargs)