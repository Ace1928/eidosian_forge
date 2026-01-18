import logging
import os
from botocore import BOTOCORE_ROOT
from botocore.compat import HAS_GZIP, OrderedDict, json
from botocore.exceptions import DataNotFoundError, UnknownServiceError
from botocore.utils import deep_merge
@instance_cache
def load_data_with_path(self, name):
    """Same as ``load_data`` but returns file path as second return value.

        :type name: str
        :param name: The data path, i.e ``ec2/2015-03-01/service-2``.

        :return: Tuple of the loaded data and the path to the data file
            where the data was loaded from. If no data could be found then a
            DataNotFoundError is raised.
        """
    for possible_path in self._potential_locations(name):
        found = self.file_loader.load_file(possible_path)
        if found is not None:
            return (found, possible_path)
    raise DataNotFoundError(data_path=name)