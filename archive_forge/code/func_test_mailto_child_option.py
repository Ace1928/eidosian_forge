from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_mailto_child_option(self):
    """Make sure that child_submit_to is used."""
    b = branch.Branch.open('branch')
    b.get_config_stack().set('mail_client', 'bogus')
    parent = branch.Branch.open('parent')
    parent.get_config_stack().set('child_submit_to', 'somebody@example.org')
    self.run_bzr_error(('Bad value "bogus" for option "mail_client"',), 'send -f branch')