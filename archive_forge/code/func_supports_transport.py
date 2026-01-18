from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
def supports_transport(self, transport):
    return False