import pytest
import decimal
import datetime
import pyarrow as pa
from pyarrow import fs
from pyarrow.tests import util
def check_example_file(orc_path, expected_df, need_fix=False):
    """
    Check a ORC file against the expected columns dictionary.
    """
    from pyarrow import orc
    orc_file = orc.ORCFile(orc_path)
    table = orc_file.read()
    assert isinstance(table, pa.Table)
    table.validate()
    orc_df = pd.DataFrame(table.to_pydict())
    assert set(expected_df.columns) == set(orc_df.columns)
    if not orc_df.columns.equals(expected_df.columns):
        expected_df = expected_df.reindex(columns=orc_df.columns)
    if need_fix:
        fix_example_values(orc_df, expected_df)
    check_example_values(orc_df, expected_df)
    json_pos = 0
    for i in range(orc_file.nstripes):
        batch = orc_file.read_stripe(i)
        check_example_values(pd.DataFrame(batch.to_pydict()), expected_df, start=json_pos, stop=json_pos + len(batch))
        json_pos += len(batch)
    assert json_pos == orc_file.nrows