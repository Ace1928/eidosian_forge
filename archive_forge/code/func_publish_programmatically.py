import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def publish_programmatically(source_class, source, source_path, destination_class, destination, destination_path, reader, reader_name, parser, parser_name, writer, writer_name, settings, settings_spec, settings_overrides, config_section, enable_exit_status):
    """
    Set up & run a `Publisher` for custom programmatic use.  Return the
    encoded string output and the Publisher object.

    Applications should not need to call this function directly.  If it does
    seem to be necessary to call this function directly, please write to the
    Docutils-develop mailing list
    <http://docutils.sf.net/docs/user/mailing-lists.html#docutils-develop>.

    Parameters:

    * `source_class` **required**: The class for dynamically created source
      objects.  Typically `io.FileInput` or `io.StringInput`.

    * `source`: Type depends on `source_class`:

      - If `source_class` is `io.FileInput`: Either a file-like object
        (must have 'read' and 'close' methods), or ``None``
        (`source_path` is opened).  If neither `source` nor
        `source_path` are supplied, `sys.stdin` is used.

      - If `source_class` is `io.StringInput` **required**: The input
        string, either an encoded 8-bit string (set the
        'input_encoding' setting to the correct encoding) or a Unicode
        string (set the 'input_encoding' setting to 'unicode').

    * `source_path`: Type depends on `source_class`:

      - `io.FileInput`: Path to the input file, opened if no `source`
        supplied.

      - `io.StringInput`: Optional.  Path to the file or object that produced
        `source`.  Only used for diagnostic output.

    * `destination_class` **required**: The class for dynamically created
      destination objects.  Typically `io.FileOutput` or `io.StringOutput`.

    * `destination`: Type depends on `destination_class`:

      - `io.FileOutput`: Either a file-like object (must have 'write' and
        'close' methods), or ``None`` (`destination_path` is opened).  If
        neither `destination` nor `destination_path` are supplied,
        `sys.stdout` is used.

      - `io.StringOutput`: Not used; pass ``None``.

    * `destination_path`: Type depends on `destination_class`:

      - `io.FileOutput`: Path to the output file.  Opened if no `destination`
        supplied.

      - `io.StringOutput`: Path to the file or object which will receive the
        output; optional.  Used for determining relative paths (stylesheets,
        source links, etc.).

    * `reader`: A `docutils.readers.Reader` object.

    * `reader_name`: Name or alias of the Reader class to be instantiated if
      no `reader` supplied.

    * `parser`: A `docutils.parsers.Parser` object.

    * `parser_name`: Name or alias of the Parser class to be instantiated if
      no `parser` supplied.

    * `writer`: A `docutils.writers.Writer` object.

    * `writer_name`: Name or alias of the Writer class to be instantiated if
      no `writer` supplied.

    * `settings`: A runtime settings (`docutils.frontend.Values`) object, for
      dotted-attribute access to runtime settings.  It's the end result of the
      `SettingsSpec`, config file, and option processing.  If `settings` is
      passed, it's assumed to be complete and no further setting/config/option
      processing is done.

    * `settings_spec`: A `docutils.SettingsSpec` subclass or object.  Provides
      extra application-specific settings definitions independently of
      components.  In other words, the application becomes a component, and
      its settings data is processed along with that of the other components.
      Used only if no `settings` specified.

    * `settings_overrides`: A dictionary containing application-specific
      settings defaults that override the defaults of other components.
      Used only if no `settings` specified.

    * `config_section`: A string, the name of the configuration file section
      for this application.  Overrides the ``config_section`` attribute
      defined by `settings_spec`.  Used only if no `settings` specified.

    * `enable_exit_status`: Boolean; enable exit status at end of processing?
    """
    pub = Publisher(reader, parser, writer, settings=settings, source_class=source_class, destination_class=destination_class)
    pub.set_components(reader_name, parser_name, writer_name)
    pub.process_programmatic_settings(settings_spec, settings_overrides, config_section)
    pub.set_source(source, source_path)
    pub.set_destination(destination, destination_path)
    output = pub.publish(enable_exit_status=enable_exit_status)
    return (output, pub)