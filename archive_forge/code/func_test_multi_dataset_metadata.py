import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
@pytest.mark.pandas
def test_multi_dataset_metadata(tempdir):
    filenames = ['ARROW-1983-dataset.0', 'ARROW-1983-dataset.1']
    metapath = str(tempdir / '_metadata')
    df = pd.DataFrame({'one': [1, 2, 3], 'two': [-1, -2, -3], 'three': [[1, 2], [2, 3], [3, 4]]})
    table = pa.Table.from_pandas(df)
    _meta = None
    for filename in filenames:
        meta = []
        pq.write_table(table, str(tempdir / filename), metadata_collector=meta)
        meta[0].set_file_path(filename)
        if _meta is None:
            _meta = meta[0]
        else:
            _meta.append_row_groups(meta[0])
    with open(metapath, 'wb') as f:
        _meta.write_metadata_file(f)
    meta = pq.read_metadata(metapath)
    md = meta.to_dict()
    _md = _meta.to_dict()
    for key in _md:
        if key != 'serialized_size':
            assert _md[key] == md[key]
    assert _md['num_columns'] == 3
    assert _md['num_rows'] == 6
    assert _md['num_row_groups'] == 2
    assert _md['serialized_size'] == 0
    assert md['serialized_size'] > 0