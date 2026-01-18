from __future__ import annotations
import urllib.parse as parse
from typing import Any
from streamlit import util
from streamlit.constants import (
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
@gather_metrics('experimental_get_query_params')
def get_query_params() -> dict[str, list[str]]:
    """Return the query parameters that is currently showing in the browser's URL bar.

    Returns
    -------
    dict
      The current query parameters as a dict. "Query parameters" are the part of the URL that comes
      after the first "?".

    Example
    -------
    Let's say the user's web browser is at
    `http://localhost:8501/?show_map=True&selected=asia&selected=america`.
    Then, you can get the query parameters using the following:

    >>> import streamlit as st
    >>>
    >>> st.experimental_get_query_params()
    {"show_map": ["True"], "selected": ["asia", "america"]}

    Note that the values in the returned dict are *always* lists. This is
    because we internally use Python's urllib.parse.parse_qs(), which behaves
    this way. And this behavior makes sense when you consider that every item
    in a query string is potentially a 1-element array.

    """
    ctx = get_script_run_ctx()
    if ctx is None:
        return {}
    ctx.mark_experimental_query_params_used()
    return util.exclude_keys_in_dict(parse.parse_qs(ctx.query_string, keep_blank_values=True), keys_to_exclude=EMBED_QUERY_PARAMS_KEYS)