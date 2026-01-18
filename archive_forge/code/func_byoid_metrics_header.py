import platform
from google.auth import version
def byoid_metrics_header(metrics_options):
    header = '{} {}'.format(python_and_auth_lib_version(), BYOID_HEADER_SECTION)
    for key, value in metrics_options.items():
        header = '{} {}/{}'.format(header, key, value)
    return header