importing any other Kivy modules. Ideally, this means setting them right at
from collections import OrderedDict
from os import environ
from os.path import exists
from weakref import ref
from kivy import kivy_config_fn
from kivy.compat import PY2, string_types
from kivy.logger import Logger, logger_config_update
from kivy.utils import platform
def adddefaultsection(self, section):
    """Add a section if the section is missing.
        """
    assert '_' not in section
    if self.has_section(section):
        return
    self.add_section(section)