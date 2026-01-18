from . import measure
from . import select
from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import warnings
def remove_rare_genes(data, *extra_data, cutoff=0, min_cells=5):
    warnings.warn('`scprep.filter.remove_rare_genes` is deprecated. Use `scprep.filter.filter_rare_genes` instead.', DeprecationWarning)
    return filter_rare_genes(data, *extra_data, cutoff=cutoff, min_cells=min_cells)