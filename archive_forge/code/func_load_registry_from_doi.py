import os
import time
import contextlib
from pathlib import Path
import shlex
import shutil
from .hashes import hash_matches, file_hash
from .utils import (
from .downloaders import DOIDownloader, choose_downloader, doi_to_repository
def load_registry_from_doi(self):
    """
        Populate the registry using the data repository API

        Fill the registry with all the files available in the data repository,
        along with their hashes. It will make a request to the data repository
        API to retrieve this information. No file is downloaded during this
        process.

        .. important::

            This method is intended to be used only when the ``base_url`` is
            a DOI.
        """
    downloader = choose_downloader(self.base_url)
    if not isinstance(downloader, DOIDownloader):
        raise ValueError(f"Invalid base_url '{self.base_url}': " + 'Pooch.load_registry_from_doi is only implemented for DOIs')
    doi = self.base_url.replace('doi:', '')
    repository = doi_to_repository(doi)
    return repository.populate_registry(self)