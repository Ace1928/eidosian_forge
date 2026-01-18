from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
import simplejson as json
from osc_lib import exceptions
from osc_lib.i18n import _
def find_one(self, path, **kwargs):
    """Find a resource by name or ID

        :param string path:
            The API-specific portion of the URL path
        :returns:
            resource dict
        """
    bulk_list = self.find_bulk(path, **kwargs)
    num_bulk = len(bulk_list)
    if num_bulk == 0:
        msg = _('none found')
        raise exceptions.NotFound(404, msg)
    elif num_bulk > 1:
        msg = _('many found')
        raise RuntimeError(msg)
    return bulk_list[0]