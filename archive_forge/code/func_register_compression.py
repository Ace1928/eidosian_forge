from zipfile import ZipFile
import fsspec.utils
from fsspec.spec import AbstractBufferedFile
def register_compression(name, callback, extensions, force=False):
    """Register an "inferable" file compression type.

    Registers transparent file compression type for use with fsspec.open.
    Compression can be specified by name in open, or "infer"-ed for any files
    ending with the given extensions.

    Args:
        name: (str) The compression type name. Eg. "gzip".
        callback: A callable of form (infile, mode, **kwargs) -> file-like.
            Accepts an input file-like object, the target mode and kwargs.
            Returns a wrapped file-like object.
        extensions: (str, Iterable[str]) A file extension, or list of file
            extensions for which to infer this compression scheme. Eg. "gz".
        force: (bool) Force re-registration of compression type or extensions.

    Raises:
        ValueError: If name or extensions already registered, and not force.

    """
    if isinstance(extensions, str):
        extensions = [extensions]
    if name in compr and (not force):
        raise ValueError(f'Duplicate compression registration: {name}')
    for ext in extensions:
        if ext in fsspec.utils.compressions and (not force):
            raise ValueError(f'Duplicate compression file extension: {ext} ({name})')
    compr[name] = callback
    for ext in extensions:
        fsspec.utils.compressions[ext] = name