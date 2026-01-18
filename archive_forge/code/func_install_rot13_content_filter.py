import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def install_rot13_content_filter(self, pattern):
    original_registry = filters._reset_registry()

    def restore_registry():
        filters._reset_registry(original_registry)
    self.addCleanup(restore_registry)

    def rot13(chunks, context=None):
        return [codecs.encode(chunk.decode('ascii'), 'rot13').encode('ascii') for chunk in chunks]
    rot13filter = filters.ContentFilter(rot13, rot13)
    filters.filter_stacks_registry.register('rot13', {'yes': [rot13filter]}.get)
    os.mkdir(self.test_home_dir + '/.bazaar')
    rules_filename = self.test_home_dir + '/.bazaar/rules'
    with open(rules_filename, 'wb') as f:
        f.write(b'[name %s]\nrot13=yes\n' % (pattern,))

    def uninstall_rules():
        os.remove(rules_filename)
        rules.reset_rules()
    self.addCleanup(uninstall_rules)
    rules.reset_rules()