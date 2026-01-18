import os
import sys
import time
import shutil
import platform
import tempfile
import unittest
import multiprocessing
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
def test_download_object_and_overwrite(self):
    tmppath = self.make_tmp_file()
    container = self.driver.create_container('test6')
    obj = container.upload_object(tmppath, 'test')
    destination_path = tmppath + '.temp'
    result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
    self.assertTrue(result)
    try:
        self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
    except LibcloudError:
        pass
    else:
        self.fail('Exception was not thrown')
    result = self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=True, delete_on_failure=True)
    self.assertTrue(result)
    obj.delete()
    container.delete()
    self.remove_tmp_file(tmppath)
    os.unlink(destination_path)