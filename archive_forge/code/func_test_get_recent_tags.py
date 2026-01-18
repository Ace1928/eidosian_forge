import datetime
import os
import re
import shutil
import tempfile
import time
import unittest
from typing import ClassVar, Dict, List, Optional, Tuple
from dulwich.contrib import release_robot
from ..repo import Repo
from ..tests.utils import make_commit, make_tag
def test_get_recent_tags(self):
    """Test get recent tags."""
    tags = release_robot.get_recent_tags(self.projdir)
    for tag, metadata in tags:
        tag = tag.encode('utf-8')
        test_data = self.tag_test_data[tag]
        self.assertEqual(metadata[0], gmtime_to_datetime(test_data[0]))
        self.assertEqual(metadata[1].encode('utf-8'), test_data[1])
        self.assertEqual(metadata[2].encode('utf-8'), self.committer)
        tag_obj = test_data[2]
        if not tag_obj:
            continue
        self.assertEqual(metadata[3][0], gmtime_to_datetime(tag_obj[0]))
        self.assertEqual(metadata[3][1].encode('utf-8'), tag_obj[1])
        self.assertEqual(metadata[3][2].encode('utf-8'), tag)