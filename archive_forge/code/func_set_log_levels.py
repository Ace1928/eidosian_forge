from cinderclient import api_versions
from cinderclient import base
@api_versions.wraps('3.32')
def set_log_levels(self, level, binary, server, prefix):
    """Set log level for services."""
    body = {'level': level, 'binary': binary, 'server': server, 'prefix': prefix}
    return self._update('/os-services/set-log', body)