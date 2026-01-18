from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def make_stacked_on_matching(self, source):
    if source.repository.supports_rich_root():
        if source.repository._format.supports_chks:
            format = '2a'
        else:
            format = '1.9-rich-root'
    else:
        format = '1.9'
    return self.make_branch('stack-on', format)