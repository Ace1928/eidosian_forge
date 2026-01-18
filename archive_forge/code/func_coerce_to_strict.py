import json
import decimal
import datetime
import warnings
from pathlib import Path
from plotly.io._utils import validate_coerce_fig_to_dict, validate_coerce_output_type
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
from_json_plotly requires a string or bytes argument but received value of type {typ}
def coerce_to_strict(const):
    """
    This is used to ultimately *encode* into strict JSON, see `encode`

    """
    if const in ('Infinity', '-Infinity', 'NaN'):
        return None
    else:
        return const