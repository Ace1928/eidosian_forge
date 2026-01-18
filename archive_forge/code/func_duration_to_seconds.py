from __future__ import annotations
import math
from datetime import timedelta
from typing import Any, Literal, overload
from streamlit import config
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_cache import populate_hash_if_needed
def duration_to_seconds(ttl: float | timedelta | str | None, *, coerce_none_to_inf: bool=True) -> float | None:
    """
    Convert a ttl value to a float representing "number of seconds".
    """
    if coerce_none_to_inf and ttl is None:
        return math.inf
    if isinstance(ttl, timedelta):
        return ttl.total_seconds()
    if isinstance(ttl, str):
        import numpy as np
        import pandas as pd
        try:
            out: float = pd.Timedelta(ttl).total_seconds()
        except ValueError as ex:
            raise BadDurationStringError(ttl) from ex
        if np.isnan(out):
            raise BadDurationStringError(ttl)
        return out
    return ttl