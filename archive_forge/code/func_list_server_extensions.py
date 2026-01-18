from __future__ import annotations
import logging
import os
import sys
import typing as t
from jupyter_core.application import JupyterApp
from jupyter_core.paths import ENV_CONFIG_PATH, SYSTEM_CONFIG_PATH, jupyter_config_dir
from tornado.log import LogFormatter
from traitlets import Bool
from jupyter_server._version import __version__
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager, ExtensionPackage
def list_server_extensions(self) -> None:
    """List all enabled and disabled server extensions, by config path

        Enabled extensions are validated, potentially generating warnings.
        """
    configurations = ({'user': True, 'sys_prefix': False}, {'user': False, 'sys_prefix': True}, {'user': False, 'sys_prefix': False})
    for option in configurations:
        config_dir = _get_config_dir(**option)
        self.log.info(f'Config dir: {config_dir}')
        write_dir = 'jupyter_server_config.d'
        config_manager = ExtensionConfigManager(read_config_path=[config_dir], write_config_dir=os.path.join(config_dir, write_dir))
        jpserver_extensions = config_manager.get_jpserver_extensions()
        for name, enabled in jpserver_extensions.items():
            self.log.info(f'    {name} {(GREEN_ENABLED if enabled else RED_DISABLED)}')
            try:
                self.log.info(f'    - Validating {name}...')
                extension = ExtensionPackage(name=name, enabled=enabled)
                if not extension.validate():
                    msg = 'validation failed'
                    raise ValueError(msg)
                version = extension.version
                self.log.info(f'      {name} {version} {GREEN_OK}')
            except Exception as err:
                exc_info = False
                if int(self.log_level) <= logging.DEBUG:
                    exc_info = True
                self.log.warning(f'      {RED_X} {err}', exc_info=exc_info)
        self.log.info('')