import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def report_Exception(self, error):
    if isinstance(error, utils.SystemMessage):
        self.report_SystemMessage(error)
    elif isinstance(error, UnicodeEncodeError):
        self.report_UnicodeError(error)
    elif isinstance(error, io.InputError):
        self._stderr.write('Unable to open source file for reading:\n  %s\n' % ErrorString(error))
    elif isinstance(error, io.OutputError):
        self._stderr.write('Unable to open destination file for writing:\n  %s\n' % ErrorString(error))
    else:
        print('%s' % ErrorString(error), file=self._stderr)
        print('Exiting due to error.  Use "--traceback" to diagnose.\nPlease report errors to <docutils-users@lists.sf.net>.\nInclude "--traceback" output, Docutils version (%s%s),\nPython version (%s), your OS type & version, and the\ncommand line used.' % (__version__, docutils.__version_details__ and ' [%s]' % docutils.__version_details__ or '', sys.version.split()[0]), file=self._stderr)