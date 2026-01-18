import configparser
from os import path
from typing import Dict, Optional
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.errors import ThemeError
from sphinx.locale import __
from sphinx.util import logging
def load_builtin_themes(self, config: Config) -> None:
    """Load built-in themes."""
    self.themes['manual'] = BuiltInTheme('manual', config)
    self.themes['howto'] = BuiltInTheme('howto', config)