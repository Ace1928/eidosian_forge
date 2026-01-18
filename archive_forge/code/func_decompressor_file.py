import io
import zlib
from joblib.backports import LooseVersion
def decompressor_file(self, fileobj):
    """Returns an instance of a decompressor file object."""
    self._check_versions()
    return self.fileobj_factory(fileobj, 'rb')