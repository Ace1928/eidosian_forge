import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def get_csv_data(self):
    """
        Get CSV data from the directive content, from an external
        file, or from a URL reference.
        """
    encoding = self.options.get('encoding', self.state.document.settings.input_encoding)
    error_handler = self.state.document.settings.input_encoding_error_handler
    if self.content:
        if 'file' in self.options or 'url' in self.options:
            error = self.state_machine.reporter.error('"%s" directive may not both specify an external file and have content.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)
        source = self.content.source(0)
        csv_data = self.content
    elif 'file' in self.options:
        if 'url' in self.options:
            error = self.state_machine.reporter.error('The "file" and "url" options may not be simultaneously specified for the "%s" directive.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)
        source_dir = os.path.dirname(os.path.abspath(self.state.document.current_source))
        source = os.path.normpath(os.path.join(source_dir, self.options['file']))
        source = utils.relative_path(None, source)
        try:
            self.state.document.settings.record_dependencies.add(source)
            csv_file = io.FileInput(source_path=source, encoding=encoding, error_handler=error_handler)
            csv_data = csv_file.read().splitlines()
        except IOError as error:
            severe = self.state_machine.reporter.severe('Problems with "%s" directive path:\n%s.' % (self.name, SafeString(error)), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(severe)
    elif 'url' in self.options:
        import urllib.request, urllib.error, urllib.parse
        source = self.options['url']
        try:
            csv_text = urllib.request.urlopen(source).read()
        except (urllib.error.URLError, IOError, OSError, ValueError) as error:
            severe = self.state_machine.reporter.severe('Problems with "%s" directive URL "%s":\n%s.' % (self.name, self.options['url'], SafeString(error)), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(severe)
        csv_file = io.StringInput(source=csv_text, source_path=source, encoding=encoding, error_handler=self.state.document.settings.input_encoding_error_handler)
        csv_data = csv_file.read().splitlines()
    else:
        error = self.state_machine.reporter.warning('The "%s" directive requires content; none supplied.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
        raise SystemMessagePropagation(error)
    return (csv_data, source)