import time
from io import BytesIO
from ... import errors as bzr_errors
from ... import tests
from ...tests.features import Feature, ModuleAvailableFeature
from .. import import_dulwich
def delete_entry(self, path):
    """This will delete files or symlinks at the given location."""
    self.commit_info.append(b'D %s\n' % (self._encode_path(path),))