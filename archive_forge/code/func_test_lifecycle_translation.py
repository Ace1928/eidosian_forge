from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
import posixpath
from unittest import mock
from xml.dom.minidom import parseString
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils.translation_helper import LifecycleTranslation
from gslib.utils import shim_util
def test_lifecycle_translation(self):
    """Tests lifecycle translation for various formats."""
    json_text = self.lifecycle_doc_without_storage_class_fields
    entries_list = LifecycleTranslation.JsonLifecycleToMessage(json_text)
    boto_lifecycle = LifecycleTranslation.BotoLifecycleFromMessage(entries_list)
    converted_entries_list = LifecycleTranslation.BotoLifecycleToMessage(boto_lifecycle)
    converted_json_text = LifecycleTranslation.JsonLifecycleFromMessage(converted_entries_list)
    self.assertEqual(json.loads(json_text), json.loads(converted_json_text))