import time
from io import BytesIO
from ... import errors as bzr_errors
from ... import tests
from ...tests.features import Feature, ModuleAvailableFeature
from .. import import_dulwich
def set_symlink(self, path, content):
    """Create or update symlink at a given path."""
    mark = self._create_blob(self._encode_path(content))
    mode = b'120000'
    self.commit_info.append(b'M %s :%d %s\n' % (mode, mark, self._encode_path(path)))