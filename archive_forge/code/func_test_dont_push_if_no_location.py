import os
from .... import tests
from ... import upload
from .. import cmds
def test_dont_push_if_no_location(self):
    self.assertPathDoesNotExist('target')
    self.build_tree(['b'])
    self.wt.add(['b'])
    self.wt.commit('two')
    self.assertPathDoesNotExist('target')