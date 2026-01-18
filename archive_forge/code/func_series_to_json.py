from traitlets import TraitError, TraitType
import numpy as np
import pandas as pd
import warnings
import datetime as dt
import six
def series_to_json(value, obj):
    return value.to_dict()