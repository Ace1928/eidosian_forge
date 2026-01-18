import configparser
from os import path
from typing import Dict, Optional
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging
def find_user_theme(self, name: str) -> Optional[Theme]:
    """Find a theme named as *name* from latex_theme_path."""
    for theme_path in self.theme_paths:
        config_path = path.join(theme_path, name, 'theme.conf')
        if path.isfile(config_path):
            try:
                return UserTheme(name, config_path)
            except ThemeError as exc:
                logger.warning(exc)
    return None