from toolz import curried
import uuid
from weakref import WeakValueDictionary
from typing import (
from altair.utils._importers import import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.data import DataType, ToValuesReturnType, MaxRowsError
from altair.vegalite.data import default_data_transformer
def compile_to_vegafusion_chart_state(vegalite_spec: dict, local_tz: str) -> 'ChartState':
    """Compile a Vega-Lite spec to a VegaFusion ChartState

    Note: This function should only be called on a Vega-Lite spec
    that was generated with the "vegafusion" data transformer enabled.
    In particular, this spec may contain references to extract datasets
    using table:// prefixed URLs.

    Parameters
    ----------
    vegalite_spec: dict
        A Vega-Lite spec that was generated from an Altair chart with
        the "vegafusion" data transformer enabled
    local_tz: str
        Local timezone name (e.g. 'America/New_York')

    Returns
    -------
    ChartState
        A VegaFusion ChartState object
    """
    from altair import vegalite_compilers, data_transformers
    vf = import_vegafusion()
    compiler = vegalite_compilers.get()
    if compiler is None:
        raise ValueError('No active vega-lite compiler plugin found')
    vega_spec = compiler(vegalite_spec)
    inline_tables = get_inline_tables(vega_spec)
    row_limit = data_transformers.options.get('max_rows', None)
    chart_state = vf.runtime.new_chart_state(vega_spec, local_tz=local_tz, inline_datasets=inline_tables, row_limit=row_limit)
    handle_row_limit_exceeded(row_limit, chart_state.get_warnings())
    return chart_state