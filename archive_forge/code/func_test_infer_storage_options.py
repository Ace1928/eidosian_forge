from __future__ import annotations
import io
import os
import pathlib
import pytest
from fsspec.utils import (
def test_infer_storage_options():
    so = infer_storage_options('/mnt/datasets/test.csv')
    assert so.pop('protocol') == 'file'
    assert so.pop('path') == '/mnt/datasets/test.csv'
    assert not so
    assert infer_storage_options('./test.csv')['path'] == './test.csv'
    assert infer_storage_options('../test.csv')['path'] == '../test.csv'
    so = infer_storage_options('C:\\test.csv')
    assert so.pop('protocol') == 'file'
    assert so.pop('path') == 'C:\\test.csv'
    assert not so
    assert infer_storage_options('d:\\test.csv')['path'] == 'd:\\test.csv'
    assert infer_storage_options('\\test.csv')['path'] == '\\test.csv'
    assert infer_storage_options('.\\test.csv')['path'] == '.\\test.csv'
    assert infer_storage_options('test.csv')['path'] == 'test.csv'
    so = infer_storage_options('hdfs://username:pwd@Node:123/mnt/datasets/test.csv?q=1#fragm', inherit_storage_options={'extra': 'value'})
    assert so.pop('protocol') == 'hdfs'
    assert so.pop('username') == 'username'
    assert so.pop('password') == 'pwd'
    assert so.pop('host') == 'Node'
    assert so.pop('port') == 123
    assert so.pop('path') == '/mnt/datasets/test.csv#fragm'
    assert so.pop('url_query') == 'q=1'
    assert so.pop('url_fragment') == 'fragm'
    assert so.pop('extra') == 'value'
    assert not so
    so = infer_storage_options('hdfs://User-name@Node-name.com/mnt/datasets/test.csv')
    assert so.pop('username') == 'User-name'
    assert so.pop('host') == 'Node-name.com'
    u = 'http://127.0.0.1:8080/test.csv'
    assert infer_storage_options(u) == {'protocol': 'http', 'path': u}
    for protocol in ['s3', 'gcs', 'gs']:
        options = infer_storage_options('%s://Bucket-name.com/test.csv' % protocol)
        assert options['path'] == 'Bucket-name.com/test.csv'
    with pytest.raises(KeyError):
        infer_storage_options('file:///bucket/file.csv', {'path': 'collide'})
    with pytest.raises(KeyError):
        infer_storage_options('hdfs:///bucket/file.csv', {'protocol': 'collide'})