import os
import sys
import ftplib
import warnings
from .utils import parse_url
def doi_to_repository(doi):
    """
    Instantiate a data repository instance from a given DOI.

    This function implements the chain of responsibility dispatch
    to the correct data repository class.

    Parameters
    ----------
    doi : str
        The DOI of the archive.

    Returns
    -------
    data_repository : DataRepository
        The data repository object
    """
    if doi[-1] == '/':
        doi = doi[:-1]
    repositories = [FigshareRepository, ZenodoRepository, DataverseRepository]
    archive_url = doi_to_url(doi)
    data_repository = None
    for repo in repositories:
        if data_repository is None:
            data_repository = repo.initialize(archive_url=archive_url, doi=doi)
    if data_repository is None:
        repository = parse_url(archive_url)['netloc']
        raise ValueError(f"Invalid data repository '{repository}'. To request or contribute support for this repository, please open an issue at https://github.com/fatiando/pooch/issues")
    return data_repository