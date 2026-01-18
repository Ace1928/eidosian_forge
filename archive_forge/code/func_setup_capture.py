from .... import config, msgeditor
from ....tests import TestCaseWithTransport
from ... import commitfromnews
def setup_capture(self):
    commitfromnews.register()
    msgeditor.hooks.install_named_hook('commit_message_template', self.capture_template, 'commitfromnews test template')
    self.messages = []
    self.commits = []