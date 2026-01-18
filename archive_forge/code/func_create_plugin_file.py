import os
import sys
from breezy import branch, osutils, registry, tests
def create_plugin_file(self, contents):
    """Create a file to be used as a plugin.

        This is created in a temporary directory, so that we
        are sure that it doesn't start in the plugin path.
        """
    os.mkdir('tmp')
    plugin_name = 'bzr_plugin_a_{}'.format(osutils.rand_chars(4))
    with open('tmp/' + plugin_name + '.py', 'wb') as f:
        f.write(contents)
    return plugin_name