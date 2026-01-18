import pytest
import pyarrow as pa
import pyarrow.compute as pc
from .test_extension_type import IntegerType
def test_join_extension_array_column():
    storage = pa.array([1, 2, 3], type=pa.int64())
    ty = IntegerType()
    ext_array = pa.ExtensionArray.from_storage(ty, storage)
    dict_array = pa.DictionaryArray.from_arrays(pa.array([0, 2, 1]), pa.array(['a', 'b', 'c']))
    t1 = pa.table({'colA': [1, 2, 6], 'colB': ext_array, 'colVals': ext_array})
    t2 = pa.table({'colA': [99, 2, 1], 'colC': ext_array})
    t3 = pa.table({'colA': [99, 2, 1], 'colC': ext_array, 'colD': dict_array})
    result = _perform_join('left outer', t1, ['colA'], t2, ['colA'])
    assert result['colVals'] == pa.chunked_array(ext_array)
    result = _perform_join('left outer', t1, ['colB'], t2, ['colC'])
    assert result['colB'] == pa.chunked_array(ext_array)
    result = _perform_join('left outer', t1, ['colA'], t3, ['colA'])
    assert result['colVals'] == pa.chunked_array(ext_array)
    result = _perform_join('left outer', t1, ['colB'], t3, ['colC'])
    assert result['colB'] == pa.chunked_array(ext_array)