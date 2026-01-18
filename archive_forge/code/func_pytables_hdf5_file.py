import pytest
import pandas as pd
import pandas._testing as tm
@pytest.fixture
def pytables_hdf5_file(tmp_path):
    """
    Use PyTables to create a simple HDF5 file.
    """
    table_schema = {'c0': tables.Time64Col(pos=0), 'c1': tables.StringCol(5, pos=1), 'c2': tables.Int64Col(pos=2)}
    t0 = 1561105000.0
    testsamples = [{'c0': t0, 'c1': 'aaaaa', 'c2': 1}, {'c0': t0 + 1, 'c1': 'bbbbb', 'c2': 2}, {'c0': t0 + 2, 'c1': 'ccccc', 'c2': 10 ** 5}, {'c0': t0 + 3, 'c1': 'ddddd', 'c2': 4294967295}]
    objname = 'pandas_test_timeseries'
    path = tmp_path / 'written_with_pytables.h5'
    with tables.open_file(path, mode='w') as f:
        t = f.create_table('/', name=objname, description=table_schema)
        for sample in testsamples:
            for key, value in sample.items():
                t.row[key] = value
            t.row.append()
    yield (path, objname, pd.DataFrame(testsamples))