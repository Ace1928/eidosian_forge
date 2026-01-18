import pytest
import pyarrow as pa
import pyarrow.compute as pc
from .test_extension_type import IntegerType
def test_joins_corner_cases():
    t1 = pa.Table.from_pydict({'colA': [1, 2, 3, 4, 5, 6], 'col2': ['a', 'b', 'c', 'd', 'e', 'f']})
    t2 = pa.Table.from_pydict({'colB': [1, 2, 3, 4, 5], 'col3': ['A', 'B', 'C', 'D', 'E']})
    with pytest.raises(pa.ArrowInvalid):
        _perform_join('left outer', t1, '', t2, '')
    with pytest.raises(TypeError):
        _perform_join('left outer', None, 'colA', t2, 'colB')
    with pytest.raises(ValueError):
        _perform_join('super mario join', t1, 'colA', t2, 'colB')