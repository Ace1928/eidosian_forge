from .. import branch, errors
from . import TestCaseWithTransport
def extract_in_checkout(self, a_branch):
    self.build_tree(['a/', 'a/b/', 'a/b/c/', 'a/b/c/d'])
    wt = a_branch.create_checkout('a', lightweight=True)
    wt.add(['b', 'b/c', 'b/c/d'], ids=[b'b-id', b'c-id', b'd-id'])
    wt.commit('added files')
    return wt.extract('b')