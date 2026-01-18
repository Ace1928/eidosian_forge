from .... import config, msgeditor
from ....tests import TestCaseWithTransport
from ... import commitfromnews
def test_fix_bug(self):
    self.setup_capture()
    self.enable_commitfromnews()
    builder = self.make_branch_builder('test')
    builder.start_series()
    orig_content = INITIAL_NEWS_CONTENT
    mod_content = b'----------------------------\ncommitfromnews release notes\n----------------------------\n\nNEXT (In development)\n---------------------\n\nIMPROVEMENTS\n~~~~~~~~~~~~\n\n* Created plugin, basic functionality of looking for NEWS and including the\n  NEWS diff.\n\n* Fixed a horrible bug. (lp:523423)\n\n'
    change_content = '\n* Fixed a horrible bug. (lp:523423)\n\n'
    builder.build_snapshot(None, [('add', ('', None, 'directory', None)), ('add', ('NEWS', b'foo-id', 'file', orig_content))], revision_id=b'BASE-id')
    builder.build_snapshot(None, [('modify', ('NEWS', mod_content))], message_callback=msgeditor.generate_commit_message_template)
    builder.finish_series()
    self.assertEqual([change_content], self.messages)
    self.assertEqual(1, len(self.commits))
    self.assertEqual('https://launchpad.net/bugs/523423 fixed', self.commits[0].revprops['bugs'])