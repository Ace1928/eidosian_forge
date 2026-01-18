import asyncio
import os
from importlib import import_module
from pathlib import Path
from posixpath import split
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Type
from unittest import TestCase, mock
from twisted.internet.defer import Deferred
from twisted.trial.unittest import SkipTest
from scrapy import Spider
from scrapy.crawler import Crawler
from scrapy.utils.boto import is_botocore_available
def mock_google_cloud_storage() -> Tuple[Any, Any, Any]:
    """Creates autospec mocks for google-cloud-storage Client, Bucket and Blob
    classes and set their proper return values.
    """
    from google.cloud.storage import Blob, Bucket, Client
    client_mock = mock.create_autospec(Client)
    bucket_mock = mock.create_autospec(Bucket)
    client_mock.get_bucket.return_value = bucket_mock
    blob_mock = mock.create_autospec(Blob)
    bucket_mock.blob.return_value = blob_mock
    return (client_mock, bucket_mock, blob_mock)