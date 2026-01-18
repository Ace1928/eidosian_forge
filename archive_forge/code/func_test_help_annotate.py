from breezy import config, tests
from breezy.urlutils import joinpath
from ..test_bedding import override_whoami
def test_help_annotate(self):
    """Annotate command exists"""
    out, err = self.run_bzr('--no-plugins annotate --help')