from traitlets import TraitError, TraitType
import numpy as np
import pandas as pd
import warnings
import datetime as dt
import six
def series_from_json(value, obj):
    return pd.Series(value)