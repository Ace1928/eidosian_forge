import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def process_programmatic_settings(self, settings_spec, settings_overrides, config_section):
    if self.settings is None:
        defaults = (settings_overrides or {}).copy()
        defaults.setdefault('traceback', True)
        self.get_settings(settings_spec=settings_spec, config_section=config_section, **defaults)