from traitlets import TraitError, TraitType
import numpy as np
import pandas as pd
import warnings
import datetime as dt
import six
def dataframe_from_json(value, obj):
    if value is None:
        return None
    else:
        return pd.DataFrame(value)