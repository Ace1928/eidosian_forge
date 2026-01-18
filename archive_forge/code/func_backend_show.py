import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.pyplot import subplots
from numpy import ndenumerate
from ....rcparams import rcParams
from .autocorrplot import plot_autocorr
from .bpvplot import plot_bpv
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
from .posteriorplot import plot_posterior
from .ppcplot import plot_ppc
from .rankplot import plot_rank
from .traceplot import plot_trace
from .violinplot import plot_violin
def backend_show(show):
    """Set default behaviour for show if not explicitly defined."""
    if show is None:
        show = rcParams['plot.matplotlib.show']
    return show