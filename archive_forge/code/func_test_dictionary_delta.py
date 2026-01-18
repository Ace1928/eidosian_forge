from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
def test_dictionary_delta(format_fixture):
    ty = pa.dictionary(pa.int8(), pa.utf8())
    data = [['foo', 'foo', None], ['foo', 'bar', 'foo'], ['foo', 'bar'], ['foo', None, 'bar', 'quux'], ['bar', 'quux']]
    batches = [pa.RecordBatch.from_arrays([pa.array(v, type=ty)], names=['dicts']) for v in data]
    batches_delta_only = batches[:4]
    schema = batches[0].schema

    def write_batches(batches, as_table=False):
        with format_fixture._get_writer(pa.MockOutputStream(), schema) as writer:
            if as_table:
                table = pa.Table.from_batches(batches)
                writer.write_table(table)
            else:
                for batch in batches:
                    writer.write_batch(batch)
            return writer.stats
    if format_fixture.is_file:
        with pytest.raises(pa.ArrowInvalid):
            write_batches(batches)
        with pytest.raises(pa.ArrowInvalid):
            write_batches(batches_delta_only)
    else:
        st = write_batches(batches)
        assert st.num_record_batches == 5
        assert st.num_dictionary_batches == 4
        assert st.num_replaced_dictionaries == 3
        assert st.num_dictionary_deltas == 0
    format_fixture.use_legacy_ipc_format = None
    format_fixture.options = pa.ipc.IpcWriteOptions(emit_dictionary_deltas=True)
    if format_fixture.is_file:
        with pytest.raises(pa.ArrowInvalid):
            write_batches(batches)
    else:
        st = write_batches(batches)
        assert st.num_record_batches == 5
        assert st.num_dictionary_batches == 4
        assert st.num_replaced_dictionaries == 1
        assert st.num_dictionary_deltas == 2
    st = write_batches(batches_delta_only)
    assert st.num_record_batches == 4
    assert st.num_dictionary_batches == 3
    assert st.num_replaced_dictionaries == 0
    assert st.num_dictionary_deltas == 2
    format_fixture.options = pa.ipc.IpcWriteOptions(unify_dictionaries=True)
    st = write_batches(batches, as_table=True)
    assert st.num_record_batches == 5
    if format_fixture.is_file:
        assert st.num_dictionary_batches == 1
        assert st.num_replaced_dictionaries == 0
        assert st.num_dictionary_deltas == 0
    else:
        assert st.num_dictionary_batches == 4
        assert st.num_replaced_dictionaries == 3
        assert st.num_dictionary_deltas == 0