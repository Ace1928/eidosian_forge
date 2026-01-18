from breezy import controldir, errors, tests
from breezy.tests import per_controldir
def make_controldir_with_repo(self):
    if not self.bzrdir_format.is_supported():
        raise tests.TestNotApplicable('Control dir format not supported')
    t = self.get_transport()
    try:
        made_control = self.make_controldir('.', format=self.bzrdir_format)
    except errors.UninitializableFormat:
        raise tests.TestNotApplicable('Control dir format not initializable')
    self.assertEqual(made_control._format, self.bzrdir_format)
    made_repo = made_control.create_repository()
    return made_control