import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def determine_wants(*args, **kwargs):
    return {'refs/heads/master': local_repo.refs['HEAD'], 'refs/tags/v1.0': local_repo.refs['refs/tags/v1.0']}