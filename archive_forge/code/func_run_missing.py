from breezy import osutils, tests
def run_missing(args, retcode=1, working_dir=None):
    out, err = self.run_bzr(['missing'] + args, retcode=retcode, working_dir=working_dir)
    self.assertEqual('', err)
    return out.splitlines()