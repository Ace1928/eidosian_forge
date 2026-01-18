import json
import decimal
import datetime
import warnings
from pathlib import Path
from plotly.io._utils import validate_coerce_fig_to_dict, validate_coerce_output_type
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
from_json_plotly requires a string or bytes argument but received value of type {typ}
def from_json_plotly(value, engine=None):
    """
    Parse JSON string using the specified JSON engine

    Parameters
    ----------
    value: str or bytes
        A JSON string or bytes object

    engine: str (default None)
        The JSON decoding engine to use. One of:
          - if "json", parse JSON using built in json module
          - if "orjson", parse using the faster orjson module, requires the orjson
            package
          - if "auto" use orjson module if available, otherwise use the json module

        If not specified, the default engine is set to the current value of
        plotly.io.json.config.default_engine.

    Returns
    -------
    dict

    See Also
    --------
    from_json_plotly : Parse JSON with plotly conventions into a dict
    """
    orjson = get_module('orjson', should_load=True)
    if not isinstance(value, (str, bytes)):
        raise ValueError('\nfrom_json_plotly requires a string or bytes argument but received value of type {typ}\n    Received value: {value}'.format(typ=type(value), value=value))
    if engine is None:
        engine = config.default_engine
    if engine == 'auto':
        if orjson is not None:
            engine = 'orjson'
        else:
            engine = 'json'
    elif engine not in ['orjson', 'json']:
        raise ValueError('Invalid json engine: %s' % engine)
    if engine == 'orjson':
        JsonConfig.validate_orjson()
        value_dict = orjson.loads(value)
    else:
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        value_dict = json.loads(value)
    return value_dict