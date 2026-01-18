import os
import subprocess
import sys
from io import BytesIO
from dulwich.repo import Repo
from ...tests import TestCaseWithTransport
from ...tests.features import PathFeature
from ..git_remote_helper import RemoteHelper, fetch, open_local_dir
from ..object_store import get_object_store
from . import FastimportFeature
def test_from_dir(self):
    self.make_branch_and_tree('.', format='git')
    open_local_dir()