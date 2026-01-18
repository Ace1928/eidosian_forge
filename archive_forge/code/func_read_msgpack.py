import gc
from . import msgpack
from .msgpack import msgpack_encoders, msgpack_decoders  # noqa: F401
from .util import force_path, FilePath, JSONInputBin, JSONOutputBin
def read_msgpack(path: FilePath, use_list: bool=True) -> JSONOutputBin:
    """Load a msgpack file.

    location (FilePath): The file path.
    use_list (bool): Don't use tuples instead of lists. Can make
        deserialization slower.
    RETURNS (JSONOutputBin): The loaded and deserialized content.
    """
    file_path = force_path(path)
    with file_path.open('rb') as f:
        gc.disable()
        msg = msgpack.load(f, raw=False, use_list=use_list)
        gc.enable()
        return msg