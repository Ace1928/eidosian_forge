import os
import sys
import ftplib
import warnings
from .utils import parse_url
def populate_registry(self, pooch):
    """
        Populate the registry using the data repository's API

        Parameters
        ----------
        pooch : Pooch
            The pooch instance that the registry will be added to.
        """
    for filedata in self.api_response.json()['data']['latestVersion']['files']:
        pooch.registry[filedata['dataFile']['filename']] = f'md5:{filedata['dataFile']['md5']}'