from __future__ import annotations
import io
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
import pytest
from fsspec.compression import compr
from fsspec.core import get_fs_token_paths, open_files
from s3fs import S3FileSystem as DaskS3FileSystem
from tlz import concat, valmap
from dask import compute
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
@pytest.fixture()
def s3_with_yellow_tripdata(s3):
    """
    Fixture with sample yellowtrip CSVs loaded into S3.

    Provides the following CSVs:

    * s3://test/nyc-taxi/2015/yellow_tripdata_2015-01.csv
    * s3://test/nyc-taxi/2014/yellow_tripdata_2015-mm.csv
      for mm from 01 - 12.
    """
    np = pytest.importorskip('numpy')
    pd = pytest.importorskip('pandas')
    data = {'VendorID': {0: 2, 1: 1, 2: 1, 3: 1, 4: 1}, 'tpep_pickup_datetime': {0: '2015-01-15 19:05:39', 1: '2015-01-10 20:33:38', 2: '2015-01-10 20:33:38', 3: '2015-01-10 20:33:39', 4: '2015-01-10 20:33:39'}, 'tpep_dropoff_datetime': {0: '2015-01-15 19:23:42', 1: '2015-01-10 20:53:28', 2: '2015-01-10 20:43:41', 3: '2015-01-10 20:35:31', 4: '2015-01-10 20:52:58'}, 'passenger_count': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}, 'trip_distance': {0: 1.59, 1: 3.3, 2: 1.8, 3: 0.5, 4: 3.0}, 'pickup_longitude': {0: -73.993896484375, 1: -74.00164794921875, 2: -73.96334075927734, 3: -74.00908660888672, 4: -73.97117614746094}, 'pickup_latitude': {0: 40.7501106262207, 1: 40.7242431640625, 2: 40.80278778076172, 3: 40.71381759643555, 4: 40.762428283691406}, 'RateCodeID': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}, 'store_and_fwd_flag': {0: 'N', 1: 'N', 2: 'N', 3: 'N', 4: 'N'}, 'dropoff_longitude': {0: -73.97478485107422, 1: -73.99441528320312, 2: -73.95182037353516, 3: -74.00432586669923, 4: -74.00418090820312}, 'dropoff_latitude': {0: 40.75061798095703, 1: 40.75910949707031, 2: 40.82441329956055, 3: 40.71998596191406, 4: 40.742652893066406}, 'payment_type': {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}, 'fare_amount': {0: 12.0, 1: 14.5, 2: 9.5, 3: 3.5, 4: 15.0}, 'extra': {0: 1.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}, 'mta_tax': {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}, 'tip_amount': {0: 3.25, 1: 2.0, 2: 0.0, 3: 0.0, 4: 0.0}, 'tolls_amount': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, 'improvement_surcharge': {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3}, 'total_amount': {0: 17.05, 1: 17.8, 2: 10.8, 3: 4.8, 4: 16.3}}
    sample = pd.DataFrame(data)
    df = sample.take(np.arange(5).repeat(10000))
    file = io.BytesIO()
    sfile = io.TextIOWrapper(file)
    df.to_csv(sfile, index=False)
    key = 'nyc-taxi/2015/yellow_tripdata_2015-01.csv'
    client = boto3.client('s3', endpoint_url='http://127.0.0.1:5555/')
    client.put_object(Bucket=test_bucket_name, Key=key, Body=file)
    key = 'nyc-taxi/2014/yellow_tripdata_2014-{:0>2d}.csv'
    for i in range(1, 13):
        file.seek(0)
        client.put_object(Bucket=test_bucket_name, Key=key.format(i), Body=file)
    yield