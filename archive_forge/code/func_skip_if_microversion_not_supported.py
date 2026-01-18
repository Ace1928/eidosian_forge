import ast
from tempest.lib.cli import output_parser
import testtools
from manilaclient import api_versions
from manilaclient import config
def skip_if_microversion_not_supported(microversion):
    """Decorator for tests that are microversion-specific."""
    if not is_microversion_supported(microversion):
        reason = 'Skipped. Test requires microversion %s that is not allowed to be used by configuration.' % microversion
        return testtools.skip(reason)
    return lambda f: f