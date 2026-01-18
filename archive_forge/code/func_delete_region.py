import abc
from keystone.common import provider_api
import keystone.conf
from keystone import exception
@abc.abstractmethod
def delete_region(self, region_id):
    """Delete an existing region.

        :raises keystone.exception.RegionNotFound: If the region doesn't exist.

        """
    raise exception.NotImplemented()