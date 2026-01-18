from .... import config, msgeditor
from ....tests import TestCaseWithTransport
from ... import commitfromnews
def test_added_NEWS(self):
    self.setup_capture()
    self.enable_commitfromnews()
    builder = self.make_branch_builder('test')
    builder.start_series()
    content = INITIAL_NEWS_CONTENT
    builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('NEWS', b'foo-id', 'file', content))], message_callback=msgeditor.generate_commit_message_template, revision_id=b'BASE-id')
    builder.finish_series()
    self.assertEqual([content.decode('utf-8')], self.messages)