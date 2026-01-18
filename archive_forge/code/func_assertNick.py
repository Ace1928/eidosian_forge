import breezy
from breezy import branch, osutils, tests
def assertNick(self, expected, working_dir='.', explicit=None, directory=None):
    cmd = ['nick']
    if directory is not None:
        cmd.extend(['--directory', directory])
    actual = self.run_bzr(cmd, working_dir=working_dir)[0][:-1]
    self.assertEqual(expected, actual)
    if explicit is not None:
        br = branch.Branch.open(working_dir)
        conf = br.get_config()
        self.assertEqual(explicit, conf.has_explicit_nickname())
        if explicit:
            self.assertEqual(expected, conf._get_explicit_nickname())