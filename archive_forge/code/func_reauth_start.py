import platform
from google.auth import version
def reauth_start():
    return '{} {}'.format(python_and_auth_lib_version(), REQUEST_TYPE_REAUTH_START)