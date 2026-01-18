import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pyarrow as pa
@st.composite
def chunked_arrays(draw, type, min_chunks=0, max_chunks=None, chunk_size=None):
    if isinstance(type, st.SearchStrategy):
        type = draw(type)
    h.assume(not pa.types.is_struct(type))
    chunk = arrays(type, size=chunk_size)
    chunks = st.lists(chunk, min_size=min_chunks, max_size=max_chunks)
    return pa.chunked_array(draw(chunks), type=type)