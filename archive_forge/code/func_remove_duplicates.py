from . import measure
from . import select
from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import warnings
def remove_duplicates(data, *extra_data, sample_labels=None):
    warnings.warn('`scprep.filter.remove_duplicates` is deprecated. Use `scprep.filter.filter_duplicates` instead.', DeprecationWarning)
    return filter_duplicates(data, *extra_data, sample_labels=sample_labels)