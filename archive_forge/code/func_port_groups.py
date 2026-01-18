from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def port_groups(self, details=False, **query):
    """Retrieve a generator of port groups.

        :param details: A boolean indicating whether the detailed information
            for every port group should be returned.
        :param dict query: Optional query parameters to be sent to restrict
            the port groups returned. Available parameters include:

            * ``address``: Only return portgroups with the specified physical
              hardware address, typically a MAC address.
            * ``fields``: A list containing one or more fields to be returned
              in the response. This may lead to some performance gain
              because other fields of the resource are not refreshed.
            * ``limit``: Requests at most the specified number of portgroups
              returned from the query.
            * ``marker``: Specifies the ID of the last-seen portgroup. Use the
              ``limit`` parameter to make an initial limited request and
              use the ID of the last-seen portgroup from the response as
              the ``marker`` value in a subsequent limited request.
            * ``node``:only return the ones associated with this specific node
              (name or UUID), or an empty set if not found.
            * ``sort_dir``: Sorts the response by the requested sort direction.
              A valid value is ``asc`` (ascending) or ``desc``
              (descending). Default is ``asc``. You can specify multiple
              pairs of sort key and sort direction query parameters. If
              you omit the sort direction in a pair, the API uses the
              natural sorting direction of the server attribute that is
              provided as the ``sort_key``.
            * ``sort_key``: Sorts the response by the this attribute value.
              Default is ``id``. You can specify multiple pairs of sort
              key and sort direction query parameters. If you omit the
              sort direction in a pair, the API uses the natural sorting
              direction of the server attribute that is provided as the
              ``sort_key``.

        :returns: A generator of port group instances.
        """
    return _portgroup.PortGroup.list(self, details=details, **query)