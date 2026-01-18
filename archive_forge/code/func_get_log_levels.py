from cinderclient import api_versions
from cinderclient import base
@api_versions.wraps('3.32')
def get_log_levels(self, binary, server, prefix):
    """Get log levels for services."""
    body = {'binary': binary, 'server': server, 'prefix': prefix}
    response = self._update('/os-services/get-log', body)
    log_levels = []
    for entry in response['log_levels']:
        entry_levels = sorted(entry['levels'].items(), key=lambda x: x[0])
        for prefix, level in entry_levels:
            log_dict = {'binary': entry['binary'], 'host': entry['host'], 'prefix': prefix, 'level': level}
            log_levels.append(LogLevel(self, log_dict, loaded=True))
    return log_levels