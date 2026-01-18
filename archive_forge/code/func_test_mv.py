import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_mv(self):
    fname1 = self.info['filename']
    fname2 = self.info['filename'] + '2'
    dirname = self.info['directory']
    self.run_bzr_decode(['mv', 'a', fname1], fail=True)
    txt = self.run_bzr_decode(['mv', 'a', fname2])
    self.assertEqual('a => %s\n' % fname2, txt)
    self.assertPathDoesNotExist('a')
    self.assertPathExists(fname2)
    self.wt = self.wt.controldir.open_workingtree()
    self.wt.commit('renamed to non-ascii')
    os.mkdir(dirname)
    self.wt.add(dirname)
    txt = self.run_bzr_decode(['mv', fname1, fname2, dirname])
    self._check_OSX_can_roundtrip(self.info['filename'])
    self.assertEqual(['{} => {}/{}'.format(fname1, dirname, fname1), '{} => {}/{}'.format(fname2, dirname, fname2)], txt.splitlines())
    newpath = '{}/{}'.format(dirname, fname2)
    txt = self.run_bzr_raw(['mv', newpath, 'a'], encoding='ascii')[0]
    self.assertPathExists('a')
    self.assertEqual(newpath.encode('ascii', 'replace') + b' => a\n', txt)