import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def list_availability_zone_names(self, unavailable=False):
    """List names of availability zones.

        :param bool unavailable: Whether or not to include unavailable zones
            in the output. Defaults to False.
        :returns: A list of availability zone names, or an empty list if the
            list could not be fetched.
        """
    try:
        zones = self.compute.availability_zones()
        ret = []
        for zone in zones:
            if zone.state['available'] or unavailable:
                ret.append(zone.name)
        return ret
    except exceptions.SDKException:
        self.log.debug('Availability zone list could not be fetched', exc_info=True)
        return []