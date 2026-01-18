import re
import shutil
import subprocess
import warnings
from io import BytesIO, TextIOWrapper
from nbconvert.utils.version import check_version
from .exceptions import ConversionException
def get_pandoc_version():
    """Gets the Pandoc version if Pandoc is installed.

    If the minimal version is not met, it will probe Pandoc for its version, cache it and return that value.
    If the minimal version is met, it will return the cached version and stop probing Pandoc
    (unless `clean_cache()` is called).

    Raises
    ------
    PandocMissing
        If pandoc is unavailable.
    """
    global __version
    if __version is None:
        if not shutil.which('pandoc'):
            raise PandocMissing()
        out = subprocess.check_output(['pandoc', '-v'])
        out_lines = out.splitlines()
        version_pattern = re.compile('^\\d+(\\.\\d+){1,}$')
        for tok in out_lines[0].decode('ascii', 'replace').split():
            if version_pattern.match(tok):
                __version = tok
                break
    return __version