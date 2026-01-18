import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
def is_builtin_path(self, path):
    """Whether a given path is within the package's data directory.

        This method can be used together with load_data_with_path(name)
        to determine if data has been loaded from a file bundled with the
        package, as opposed to a file in a separate location.

        :type path: str
        :param path: The file path to check.

        :return: Whether the given path is within the package's data directory.
        """
    path = os.path.expanduser(os.path.expandvars(path))
    return path.startswith(self.BUILTIN_DATA_PATH)