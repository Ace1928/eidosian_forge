import itertools
import re
import warnings
from ..api import APIClient
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..errors import BuildError, ImageLoadError, InvalidArgument
from ..utils import parse_repository_tag
from ..utils.json_stream import json_stream
from .resource import Collection, Model
def has_platform(self, platform):
    """
        Check whether the given platform identifier is available for this
        digest.

        Args:
            platform (str or dict): A string using the ``os[/arch[/variant]]``
                format, or a platform dictionary.

        Returns:
            (bool): ``True`` if the platform is recognized as available,
            ``False`` otherwise.

        Raises:
            :py:class:`docker.errors.InvalidArgument`
                If the platform argument is not a valid descriptor.
        """
    if platform and (not isinstance(platform, dict)):
        parts = platform.split('/')
        if len(parts) > 3 or len(parts) < 1:
            raise InvalidArgument(f'"{platform}" is not a valid platform descriptor')
        platform = {'os': parts[0]}
        if len(parts) > 2:
            platform['variant'] = parts[2]
        if len(parts) > 1:
            platform['architecture'] = parts[1]
    return normalize_platform(platform, self.client.version()) in self.attrs['Platforms']