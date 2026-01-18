import itertools
from contextlib import ExitStack
def make_branch_and_populated_tree(testcase):
    """Make a simple branch and tree.

    The tree holds some added but uncommitted files.
    """
    tree = testcase.make_branch_and_tree('t')
    testcase.build_tree_contents([('t/hello', b'hello world')])
    tree.add(['hello'], ids=[b'hello-id'])
    return tree