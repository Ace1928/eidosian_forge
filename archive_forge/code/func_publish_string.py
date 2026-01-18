import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def publish_string(source, source_path=None, destination_path=None, reader=None, reader_name='standalone', parser=None, parser_name='restructuredtext', writer=None, writer_name='pseudoxml', settings=None, settings_spec=None, settings_overrides=None, config_section=None, enable_exit_status=False):
    """
    Set up & run a `Publisher` for programmatic use with string I/O.  Return
    the encoded string or Unicode string output.

    For encoded string output, be sure to set the 'output_encoding' setting to
    the desired encoding.  Set it to 'unicode' for unencoded Unicode string
    output.  Here's one way::

        publish_string(..., settings_overrides={'output_encoding': 'unicode'})

    Similarly for Unicode string input (`source`)::

        publish_string(..., settings_overrides={'input_encoding': 'unicode'})

    Parameters: see `publish_programmatically`.
    """
    output, pub = publish_programmatically(source_class=io.StringInput, source=source, source_path=source_path, destination_class=io.StringOutput, destination=None, destination_path=destination_path, reader=reader, reader_name=reader_name, parser=parser, parser_name=parser_name, writer=writer, writer_name=writer_name, settings=settings, settings_spec=settings_spec, settings_overrides=settings_overrides, config_section=config_section, enable_exit_status=enable_exit_status)
    return output