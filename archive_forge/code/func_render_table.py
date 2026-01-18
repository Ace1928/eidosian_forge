from __future__ import annotations
import base64
import html
import logging
import os
import pathlib
import pickle
import random
import re
import string
from io import StringIO
from typing import Optional, Union
from packaging.version import Version
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
@staticmethod
def render_table(table, columns=None, hide_index=True):
    """
        Renders a table-like object as an HTML table.

        Args:
            table: Table-like object (e.g. pandas DataFrame, 2D numpy array, list of tuples).
            columns: Column names to use. If `table` doesn't have column names, this argument
                provides names for the columns. Otherwise, only the specified columns will be
                included in the output HTML table.
            hide_index: Hide index column when rendering.
        """
    import pandas as pd
    from pandas.io.formats.style import Styler
    if not isinstance(table, Styler):
        table = pd.DataFrame(table, columns=columns)

        def escape_value(x):
            return html.escape(str(x))
        if hasattr(table, 'map'):
            table = table.map(escape_value)
        else:
            table = table.applymap(escape_value)
        table = table.style
    pandas_version = Version(pd.__version__)
    styler = table.set_table_attributes('style="border-collapse:collapse"').set_table_styles([{'selector': 'table, th, td', 'props': [('border', '1px solid grey'), ('text-align', 'left'), ('padding', '5px')]}])
    if hide_index:
        rendered_table = styler.hide(axis='index').to_html() if pandas_version >= Version('1.4.0') else styler.hide_index().render()
    else:
        rendered_table = styler.to_html() if pandas_version >= Version('1.4.0') else styler.render()
    return f'<div style="max-height: 500px; overflow: scroll;">{rendered_table}</div>'