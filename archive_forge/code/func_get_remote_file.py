from math import log
import os
from os import path as op
import sys
import shutil
import time
from . import appdata_dir, resource_dirs
from . import StdoutProgressIndicator, urlopen
def get_remote_file(fname, directory=None, force_download=False, auto=True):
    """Get a the filename for the local version of a file from the web

    Parameters
    ----------
    fname : str
        The relative filename on the remote data repository to download.
        These correspond to paths on
        ``https://github.com/imageio/imageio-binaries/``.
    directory : str | None
        The directory where the file will be cached if a download was
        required to obtain the file. By default, the appdata directory
        is used. This is also the first directory that is checked for
        a local version of the file. If the directory does not exist,
        it will be created.
    force_download : bool | str
        If True, the file will be downloaded even if a local copy exists
        (and this copy will be overwritten). Can also be a YYYY-MM-DD date
        to ensure a file is up-to-date (modified date of a file on disk,
        if present, is checked).
    auto : bool
        Whether to auto-download the file if its not present locally. Default
        True. If False and a download is needed, raises NeedDownloadError.

    Returns
    -------
    fname : str
        The path to the file on the local system.
    """
    _url_root = 'https://github.com/imageio/imageio-binaries/raw/master/'
    url = _url_root + fname
    nfname = op.normcase(fname)
    given_directory = directory
    directory = given_directory or appdata_dir('imageio')
    dirs = resource_dirs()
    dirs.insert(0, directory)
    for dir in dirs:
        filename = op.join(dir, nfname)
        if op.isfile(filename):
            if not force_download:
                if given_directory and given_directory != dir:
                    filename2 = os.path.join(given_directory, nfname)
                    if not op.isdir(op.dirname(filename2)):
                        os.makedirs(op.abspath(op.dirname(filename2)))
                    shutil.copy(filename, filename2)
                    return filename2
                return filename
            if isinstance(force_download, str):
                ntime = time.strptime(force_download, '%Y-%m-%d')
                ftime = time.gmtime(op.getctime(filename))
                if ftime >= ntime:
                    if given_directory and given_directory != dir:
                        filename2 = os.path.join(given_directory, nfname)
                        if not op.isdir(op.dirname(filename2)):
                            os.makedirs(op.abspath(op.dirname(filename2)))
                        shutil.copy(filename, filename2)
                        return filename2
                    return filename
                else:
                    print('File older than %s, updating...' % force_download)
                    break
    if os.getenv('IMAGEIO_NO_INTERNET', '').lower() in ('1', 'true', 'yes'):
        raise InternetNotAllowedError('Will not download resource from the internet because environment variable IMAGEIO_NO_INTERNET is set.')
    if not auto:
        raise NeedDownloadError()
    filename = op.join(directory, nfname)
    if not op.isdir(op.dirname(filename)):
        os.makedirs(op.abspath(op.dirname(filename)))
    if os.getenv('CONTINUOUS_INTEGRATION', False):
        for i in range(2):
            try:
                _fetch_file(url, filename)
                return filename
            except IOError:
                time.sleep(0.5)
        else:
            _fetch_file(url, filename)
            return filename
    else:
        _fetch_file(url, filename)
        return filename