from __future__ import annotations
import os
import urllib.parse
from .io import (
from .config import (
from .host_configs import (
from .util import (
from .util_common import (
from .docker_util import (
from .containers import (
from .ansible_util import (
from .host_profiles import (
from .inventory import (
def configure_pypi_proxy_pip(args: EnvironmentConfig, profile: HostProfile, pypi_endpoint: str, pypi_hostname: str) -> None:
    """Configure a custom index for pip based installs."""
    pip_conf_path = os.path.expanduser('~/.pip/pip.conf')
    pip_conf = '\n[global]\nindex-url = {0}\ntrusted-host = {1}\n'.format(pypi_endpoint, pypi_hostname).strip()

    def pip_conf_cleanup() -> None:
        """Remove custom pip PyPI config."""
        display.info('Removing custom PyPI config: %s' % pip_conf_path, verbosity=1)
        os.remove(pip_conf_path)
    if os.path.exists(pip_conf_path) and (not profile.config.is_managed):
        raise ApplicationError('Refusing to overwrite existing file: %s' % pip_conf_path)
    display.info('Injecting custom PyPI config: %s' % pip_conf_path, verbosity=1)
    display.info('Config: %s\n%s' % (pip_conf_path, pip_conf), verbosity=3)
    if not args.explain:
        write_text_file(pip_conf_path, pip_conf, True)
        ExitHandler.register(pip_conf_cleanup)