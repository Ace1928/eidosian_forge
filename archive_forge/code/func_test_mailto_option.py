from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def test_mailto_option(self):
    b = branch.Branch.open('branch')
    b.get_config_stack().set('mail_client', 'editor')
    self.run_bzr_error(('No mail-to address \\(--mail-to\\) or output \\(-o\\) specified',), 'send -f branch')
    b.get_config_stack().set('mail_client', 'bogus')
    self.run_send([])
    self.run_bzr_error(('Bad value "bogus" for option "mail_client"',), 'send -f branch --mail-to jrandom@example.org')
    b.get_config_stack().set('submit_to', 'jrandom@example.org')
    self.run_bzr_error(('Bad value "bogus" for option "mail_client"',), 'send -f branch')