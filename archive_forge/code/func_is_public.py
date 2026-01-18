from urllib import parse
from cinderclient import api_versions
from cinderclient import base
@property
def is_public(self):
    """
        Provide a user-friendly accessor to is_public
        """
    return self._info.get('is_public', self._info.get('is_public', 'N/A'))