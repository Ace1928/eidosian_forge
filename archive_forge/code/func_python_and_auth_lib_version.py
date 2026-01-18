import platform
from google.auth import version
def python_and_auth_lib_version():
    return 'gl-python/{} auth/{}'.format(platform.python_version(), version.__version__)