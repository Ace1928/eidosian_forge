from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
def test_continue_on_error_output(self):
    story = "\n$ brz init\n...\n$ cat >file\n<Hello\n$ brz add file\n...\n$ brz commit -m 'adding file'\n2>...\n"
    self.run_script(story)