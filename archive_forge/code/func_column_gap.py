from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Sequence, Union, cast
from typing_extensions import TypeAlias
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.runtime.metrics_util import gather_metrics
def column_gap(gap):
    if type(gap) == str:
        gap_size = gap.lower()
        valid_sizes = ['small', 'medium', 'large']
        if gap_size in valid_sizes:
            return gap_size
    raise StreamlitAPIException(f'The gap argument to st.columns must be "small", "medium", or "large". \nThe argument passed was {gap}.')