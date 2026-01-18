import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def test_clone_then_push_data(self):
    self.test_push_data_branch()
    shutil.rmtree(self.temp_d)
    local_repo = repo.Repo.init(self.temp_d, mkdir=True)
    tcp_client = client.TCPGitClient(self.server_address, port=self.port)
    remote_refs = tcp_client.fetch(self.fakerepo, local_repo)
    files = (os.path.join(self.temp_d, 'testfile'), os.path.join(self.temp_d, 'testfile2'))
    local_repo['HEAD'] = remote_refs['refs/heads/master']
    indexfile = local_repo.index_path()
    tree = local_repo['HEAD'].tree
    index.build_index_from_tree(local_repo.path, indexfile, local_repo.object_store, tree)
    for f in files:
        self.assertEqual(os.path.isfile(f), True)

    def determine_wants(*args, **kwargs):
        return {'refs/heads/master': local_repo.refs['HEAD']}
    os.mkdir(os.path.join(self.temp_d, 'test'))
    files = ('testfile11', 'testfile22', 'test/testfile33')
    i = 0
    for f in files:
        open(os.path.join(self.temp_d, f), 'w').write('DATA %s' % i)
        i += 1
    local_repo.stage(files)
    local_repo.do_commit('Test commit', 'fbo@localhost', ref='refs/heads/master')
    tcp_client.send_pack('/fakerepo', determine_wants, local_repo.generate_pack_data)