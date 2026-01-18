import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def test_describe_branch_format(self):
    for key in controldir.format_registry.keys():
        if key in controldir.format_registry.aliases():
            continue
        if controldir.format_registry.get_info(key).hidden:
            continue
        expected = None
        if key in ('dirstate', 'knit'):
            expected = 'dirstate or knit'
        elif key in ('1.14',):
            expected = '1.14'
        elif key in ('1.14-rich-root',):
            expected = '1.14-rich-root'
        self.assertBranchDescription(key, expected)