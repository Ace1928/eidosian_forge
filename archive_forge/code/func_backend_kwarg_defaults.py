from bokeh.plotting import figure
from numpy import array
from packaging import version
from ....rcparams import rcParams
from .autocorrplot import plot_autocorr
from .compareplot import plot_compare
from .densityplot import plot_density
from .distplot import plot_dist
from .elpdplot import plot_elpd
from .energyplot import plot_energy
from .essplot import plot_ess
from .forestplot import plot_forest
from .hdiplot import plot_hdi
from .kdeplot import plot_kde
from .khatplot import plot_khat
from .loopitplot import plot_loo_pit
from .mcseplot import plot_mcse
from .pairplot import plot_pair
from .parallelplot import plot_parallel
from .ppcplot import plot_ppc
from .posteriorplot import plot_posterior
from .rankplot import plot_rank
from .traceplot import plot_trace
from .violinplot import plot_violin
def backend_kwarg_defaults(*args, **kwargs):
    """Get default kwargs for backend.

    For args add a tuple with key and rcParam key pair.
    """
    defaults = {**kwargs}
    for key, arg in args:
        defaults.setdefault(key, rcParams[arg])
    for key, arg in {'toolbar_location': 'plot.bokeh.layout.toolbar_location', 'tools': 'plot.bokeh.tools', 'output_backend': 'plot.bokeh.output_backend', 'height': 'plot.bokeh.figure.height', 'width': 'plot.bokeh.figure.width'}.items():
        if key in ('height', 'width') and 'dpi' in defaults:
            continue
        defaults.setdefault(key, rcParams[arg])
    return defaults