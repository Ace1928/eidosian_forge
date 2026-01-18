import os
import time
import pytest
import srsly
from weasel.cli.remote_storage import RemoteStorage
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import validate_project_commands
@pytest.mark.parametrize('parent,child,expected', [('/tmp', '/tmp', True), ('/tmp', '/', False), ('/tmp', '/tmp/subdir', True), ('/tmp', '/tmpdir', False), ('/tmp', '/tmp/subdir/..', True), ('/tmp', '/tmp/..', False)])
def test_is_subpath_of(parent, child, expected):
    assert is_subpath_of(parent, child) == expected