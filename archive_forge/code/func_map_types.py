import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np
import pyarrow as pa
@st.composite
def map_types(draw, key_strategy=primitive_types, item_strategy=primitive_types):
    key_type = draw(key_strategy)
    h.assume(not pa.types.is_null(key_type))
    value_type = draw(item_strategy)
    return pa.map_(key_type, value_type)