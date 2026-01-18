import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def set_reader(self, reader_name, parser, parser_name):
    """Set `self.reader` by name."""
    reader_class = readers.get_reader_class(reader_name)
    self.reader = reader_class(parser, parser_name)
    self.parser = self.reader.parser