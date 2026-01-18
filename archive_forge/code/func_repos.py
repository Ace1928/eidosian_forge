import requests
from ..spec import AbstractFileSystem
from ..utils import infer_storage_options
from .memory import MemoryFile
@classmethod
def repos(cls, org_or_user, is_org=True):
    """List repo names for given org or user

        This may become the top level of the FS

        Parameters
        ----------
        org_or_user: str
            Name of the github org or user to query
        is_org: bool (default True)
            Whether the name is an organisation (True) or user (False)

        Returns
        -------
        List of string
        """
    r = requests.get(f'https://api.github.com/{['users', 'orgs'][is_org]}/{org_or_user}/repos', timeout=cls.timeout)
    r.raise_for_status()
    return [repo['name'] for repo in r.json()]