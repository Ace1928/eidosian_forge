import errno
import functools
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from typing import Tuple
from dulwich.tests import SkipTest, TestCase
from ...protocol import TCP_GIT_PORT
from ...repo import Repo
def assertReposNotEqual(self, repo1, repo2):
    refs1 = repo1.get_refs()
    objs1 = set(repo1.object_store)
    refs2 = repo2.get_refs()
    objs2 = set(repo2.object_store)
    self.assertFalse(refs1 == refs2 and objs1 == objs2)