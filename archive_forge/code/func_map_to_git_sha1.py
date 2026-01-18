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
def map_to_git_sha1(dir, bzr_revid):
    object_store = get_object_store(dir.open_repository())
    with object_store.lock_read():
        return object_store._lookup_revision_sha1(bzr_revid)