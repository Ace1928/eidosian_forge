from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from gslib.exception import CommandException
from gslib import storage_url
from gslib.exception import InvalidUrlError
from gslib.tests.testcase import base
from unittest import mock
def test_accepts_mix_of_objects_and_buckets_if_recursion_requested(self):
    urls = list(map(storage_url.StorageUrlFromString, ['gs://b1', 'gs://b/o']))
    storage_url.RaiseErrorIfUrlsAreMixOfBucketsAndObjects(urls, recursion_requested=True)