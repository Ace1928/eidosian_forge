import os
import time
import pytest
import srsly
from weasel.cli.remote_storage import RemoteStorage
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import validate_project_commands
def test_local_remote_storage():
    with make_tempdir() as d:
        filename = 'a.txt'
        content_hashes = ('aaaa', 'cccc', 'bbbb')
        for i, content_hash in enumerate(content_hashes):
            if i > 0:
                time.sleep(1)
            content = f'{content_hash} content'
            loc_file = d / 'root' / filename
            if not loc_file.parent.exists():
                loc_file.parent.mkdir(parents=True)
            with loc_file.open(mode='w') as file_:
                file_.write(content)
            remote = RemoteStorage(d / 'root', str(d / 'remote'))
            remote.push(filename, 'aaaa', content_hash)
            loc_file.unlink()
            remote.pull(filename, command_hash='aaaa', content_hash=content_hash)
            with loc_file.open(mode='r') as file_:
                assert file_.read() == content
            loc_file.unlink()
            remote.pull(filename, command_hash='aaaa')
            with loc_file.open(mode='r') as file_:
                assert file_.read() == content
            loc_file.unlink()
            remote.pull(filename, content_hash=content_hash)
            with loc_file.open(mode='r') as file_:
                assert file_.read() == content
            loc_file.unlink()
            remote.pull(filename)
            with loc_file.open(mode='r') as file_:
                assert file_.read() == content