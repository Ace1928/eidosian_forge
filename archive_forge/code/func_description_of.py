import argparse
import sys
from typing import Iterable, List, Optional
from .. import __version__
from ..universaldetector import UniversalDetector
def description_of(lines: Iterable[bytes], name: str='stdin', minimal: bool=False, should_rename_legacy: bool=False) -> Optional[str]:
    """
    Return a string describing the probable encoding of a file or
    list of strings.

    :param lines: The lines to get the encoding of.
    :type lines: Iterable of bytes
    :param name: Name of file or collection of lines
    :type name: str
    :param should_rename_legacy:  Should we rename legacy encodings to
                                  their more modern equivalents?
    :type should_rename_legacy:   ``bool``
    """
    u = UniversalDetector(should_rename_legacy=should_rename_legacy)
    for line in lines:
        line = bytearray(line)
        u.feed(line)
        if u.done:
            break
    u.close()
    result = u.result
    if minimal:
        return result['encoding']
    if result['encoding']:
        return f'{name}: {result['encoding']} with confidence {result['confidence']}'
    return f'{name}: no result'