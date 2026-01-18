import sys
import pprint
from docutils import __version__, __version_details__, SettingsSpec
from docutils import frontend, io, utils, readers, writers
from docutils.frontend import OptionParser
from docutils.transforms import Transformer
from docutils.utils.error_reporting import ErrorOutput, ErrorString
import docutils.readers.doctree
def debugging_dumps(self):
    if not self.document:
        return
    if self.settings.dump_settings:
        print('\n::: Runtime settings:', file=self._stderr)
        print(pprint.pformat(self.settings.__dict__), file=self._stderr)
    if self.settings.dump_internals:
        print('\n::: Document internals:', file=self._stderr)
        print(pprint.pformat(self.document.__dict__), file=self._stderr)
    if self.settings.dump_transforms:
        print('\n::: Transforms applied:', file=self._stderr)
        print(' (priority, transform class, pending node details, keyword args)', file=self._stderr)
        print(pprint.pformat([(priority, '%s.%s' % (xclass.__module__, xclass.__name__), pending and pending.details, kwargs) for priority, xclass, pending, kwargs in self.document.transformer.applied]), file=self._stderr)
    if self.settings.dump_pseudo_xml:
        print('\n::: Pseudo-XML:', file=self._stderr)
        print(self.document.pformat().encode('raw_unicode_escape'), file=self._stderr)