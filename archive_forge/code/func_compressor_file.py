import io
import zlib
from joblib.backports import LooseVersion
def compressor_file(self, fileobj, compresslevel=None):
    """Returns an instance of a compressor file object."""
    self._check_versions()
    if compresslevel is None:
        return self.fileobj_factory(fileobj, 'wb')
    else:
        return self.fileobj_factory(fileobj, 'wb', compression_level=compresslevel)