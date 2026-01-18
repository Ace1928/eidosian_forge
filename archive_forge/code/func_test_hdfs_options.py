from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
@pytest.mark.hdfs
def test_hdfs_options(hdfs_connection, pickle_module):
    from pyarrow.fs import HadoopFileSystem
    if not pa.have_libhdfs():
        pytest.skip('Cannot locate libhdfs')
    host, port, user = hdfs_connection
    replication = 2
    buffer_size = 64 * 1024
    default_block_size = 128 * 1024 ** 2
    uri = 'hdfs://{}:{}/?user={}&replication={}&buffer_size={}&default_block_size={}'
    hdfs1 = HadoopFileSystem(host, port, user='libhdfs', replication=replication, buffer_size=buffer_size, default_block_size=default_block_size)
    hdfs2 = HadoopFileSystem.from_uri(uri.format(host, port, 'libhdfs', replication, buffer_size, default_block_size))
    hdfs3 = HadoopFileSystem.from_uri(uri.format(host, port, 'me', replication, buffer_size, default_block_size))
    hdfs4 = HadoopFileSystem.from_uri(uri.format(host, port, 'me', replication + 1, buffer_size, default_block_size))
    hdfs5 = HadoopFileSystem(host, port)
    hdfs6 = HadoopFileSystem.from_uri('hdfs://{}:{}'.format(host, port))
    hdfs7 = HadoopFileSystem(host, port, user='localuser')
    hdfs8 = HadoopFileSystem(host, port, user='localuser', kerb_ticket='cache_path')
    hdfs9 = HadoopFileSystem(host, port, user='localuser', kerb_ticket=pathlib.Path('cache_path'))
    hdfs10 = HadoopFileSystem(host, port, user='localuser', kerb_ticket='cache_path2')
    hdfs11 = HadoopFileSystem(host, port, user='localuser', kerb_ticket='cache_path', extra_conf={'hdfs_token': 'abcd'})
    assert hdfs1 == hdfs2
    assert hdfs5 == hdfs6
    assert hdfs6 != hdfs7
    assert hdfs2 != hdfs3
    assert hdfs3 != hdfs4
    assert hdfs7 != hdfs5
    assert hdfs2 != hdfs3
    assert hdfs3 != hdfs4
    assert hdfs7 != hdfs8
    assert hdfs8 == hdfs9
    assert hdfs10 != hdfs9
    assert hdfs11 != hdfs8
    with pytest.raises(TypeError):
        HadoopFileSystem()
    with pytest.raises(TypeError):
        HadoopFileSystem.from_uri(3)
    for fs in [hdfs1, hdfs2, hdfs3, hdfs4, hdfs5, hdfs6, hdfs7, hdfs8, hdfs9, hdfs10, hdfs11]:
        assert pickle_module.loads(pickle_module.dumps(fs)) == fs
    host, port, user = hdfs_connection
    hdfs = HadoopFileSystem(host, port, user=user)
    assert hdfs.get_file_info(FileSelector('/'))
    hdfs = HadoopFileSystem.from_uri('hdfs://{}:{}/?user={}'.format(host, port, user))
    assert hdfs.get_file_info(FileSelector('/'))