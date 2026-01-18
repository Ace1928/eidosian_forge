import os
import warnings
from .array_sequence import ArraySequence
from .header import Field
from .tck import TckFile
from .tractogram import LazyTractogram, Tractogram
from .tractogram_file import ExtensionWarning
from .trk import TrkFile
def detect_format(fileobj):
    """Returns the StreamlinesFile object guessed from the file-like object.

    Parameters
    ----------
    fileobj : string or file-like object
        If string, a filename; otherwise an open file-like object pointing
        to a tractogram file (and ready to read from the beginning of the
        header)

    Returns
    -------
    tractogram_file : :class:`TractogramFile` class
        The class type guessed from the content of `fileobj`.
    """
    for format in FORMATS.values():
        try:
            if format.is_correct_format(fileobj):
                return format
        except OSError:
            pass
    if isinstance(fileobj, str):
        _, ext = os.path.splitext(fileobj)
        return FORMATS.get(ext.lower())
    return None