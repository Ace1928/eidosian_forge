import os
import sys
import ftplib
import warnings
from .utils import parse_url
def choose_downloader(url, progressbar=False):
    """
    Choose the appropriate downloader for the given URL based on the protocol.

    Parameters
    ----------
    url : str
        A URL (including protocol).
    progressbar : bool or an arbitrary progress bar object
        If True, will print a progress bar of the download to standard error
        (stderr). Requires `tqdm <https://github.com/tqdm/tqdm>`__ to be
        installed. Alternatively, an arbitrary progress bar object can be
        passed. See :ref:`custom-progressbar` for details.

    Returns
    -------
    downloader
        A downloader class, like :class:`pooch.HTTPDownloader`,
        :class:`pooch.FTPDownloader`, or :class: `pooch.SFTPDownloader`.

    Examples
    --------

    >>> downloader = choose_downloader("http://something.com")
    >>> print(downloader.__class__.__name__)
    HTTPDownloader
    >>> downloader = choose_downloader("https://something.com")
    >>> print(downloader.__class__.__name__)
    HTTPDownloader
    >>> downloader = choose_downloader("ftp://something.com")
    >>> print(downloader.__class__.__name__)
    FTPDownloader
    >>> downloader = choose_downloader("doi:DOI/filename.csv")
    >>> print(downloader.__class__.__name__)
    DOIDownloader

    """
    known_downloaders = {'ftp': FTPDownloader, 'https': HTTPDownloader, 'http': HTTPDownloader, 'sftp': SFTPDownloader, 'doi': DOIDownloader}
    parsed_url = parse_url(url)
    if parsed_url['protocol'] not in known_downloaders:
        raise ValueError(f"Unrecognized URL protocol '{parsed_url['protocol']}' in '{url}'. Must be one of {known_downloaders.keys()}.")
    downloader = known_downloaders[parsed_url['protocol']](progressbar=progressbar)
    return downloader