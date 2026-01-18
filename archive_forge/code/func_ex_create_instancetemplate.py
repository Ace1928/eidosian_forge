import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_create_instancetemplate(self, name, size, source=None, image=None, disk_type='pd-standard', disk_auto_delete=True, network='default', subnetwork=None, can_ip_forward=None, external_ip='ephemeral', internal_ip=None, service_accounts=None, on_host_maintenance=None, automatic_restart=None, preemptible=None, tags=None, metadata=None, description=None, disks_gce_struct=None, nic_gce_struct=None):
    """
        Creates an instance template in the specified project using the data
        that is included in the request. If you are creating a new template to
        update an existing instance group, your new instance template must
        use the same network or, if applicable, the same subnetwork as the
        original template.

        Scopes needed - one of the following:
        * https://www.googleapis.com/auth/cloud-platform
        * https://www.googleapis.com/auth/compute

        :param  name: The name of the node to create.
        :type   name: ``str``

        :param  size: The machine type to use.
        :type   size: ``str`` or :class:`GCENodeSize`

        :param  image: The image to use to create the node (or, if attaching
                       a persistent disk, the image used to create the disk)
        :type   image: ``str`` or :class:`GCENodeImage` or ``None``

        :keyword  network: The network to associate with the template.
        :type     network: ``str`` or :class:`GCENetwork`

        :keyword  subnetwork: The subnetwork to associate with the node.
        :type     subnetwork: ``str`` or :class:`GCESubnetwork`

        :keyword  tags: A list of tags to associate with the node.
        :type     tags: ``list`` of ``str`` or ``None``

        :keyword  metadata: Metadata dictionary for instance.
        :type     metadata: ``dict`` or ``None``

        :keyword  external_ip: The external IP address to use.  If 'ephemeral'
                               (default), a new non-static address will be
                               used.  If 'None', then no external address will
                               be used.  To use an existing static IP address,
                               a GCEAddress object should be passed in.
        :type     external_ip: :class:`GCEAddress` or ``str`` or ``None``

        :keyword  internal_ip: The private IP address to use.
        :type     internal_ip: :class:`GCEAddress` or ``str`` or ``None``

        :keyword  disk_type: Specify a pd-standard (default) disk or pd-ssd
                                for an SSD disk.
        :type     disk_type: ``str`` or :class:`GCEDiskType`

        :keyword  disk_auto_delete: Indicate that the boot disk should be
                                       deleted when the Node is deleted. Set to
                                       True by default.
        :type     disk_auto_delete: ``bool``

        :keyword  service_accounts: Specify a list of serviceAccounts when
                                       creating the instance. The format is a
                                       list of dictionaries containing email
                                       and list of scopes, e.g.
                                       [{'email':'default',
                                       'scopes':['compute', ...]}, ...]
                                       Scopes can either be full URLs or short
                                       names. If not provided, use the
                                       'default' service account email and a
                                       scope of 'devstorage.read_only'. Also
                                       accepts the aliases defined in
                                       'gcloud compute'.
        :type     service_accounts: ``list``

        :keyword  description: The description of the node (instance).
        :type     description: ``str`` or ``None``

        :keyword  can_ip_forward: Set to ``True`` to allow this node to
                                  send/receive non-matching src/dst packets.
        :type     can_ip_forward: ``bool`` or ``None``

        :keyword  disks_gce_struct: Support for passing in the GCE-specific
                                       formatted disks[] structure. No attempt
                                       is made to ensure proper formatting of
                                       the disks[] structure. Using this
                                       structure obviates the need of using
                                       other disk params like 'ex_boot_disk',
                                       etc. See the GCE docs for specific
                                       details.
        :type     disks_gce_struct: ``list`` or ``None``

        :keyword  nic_gce_struct: Support passing in the GCE-specific
                                     formatted networkInterfaces[] structure.
                                     No attempt is made to ensure proper
                                     formatting of the networkInterfaces[]
                                     data. Using this structure obviates the
                                     need of using 'external_ip' and
                                     'ex_network'.  See the GCE docs for
                                     details.
        :type     nic_gce_struct: ``list`` or ``None``

        :keyword  on_host_maintenance: Defines whether node should be
                                          terminated or migrated when host
                                          machine goes down. Acceptable values
                                          are: 'MIGRATE' or 'TERMINATE' (If
                                          not supplied, value will be reset to
                                          GCE default value for the instance
                                          type.)
        :type     ex_on_host_maintenance: ``str`` or ``None``

        :keyword  automatic_restart: Defines whether the instance should be
                                        automatically restarted when it is
                                        terminated by Compute Engine. (If not
                                        supplied, value will be set to the GCE
                                        default value for the instance type.)
        :type     automatic_restart: ``bool`` or ``None``

        :keyword  preemptible: Defines whether the instance is preemptible.
                                  (If not supplied, the instance will not be
                                  preemptible)
        :type     preemptible: ``bool`` or ``None``

        :return:  An Instance Template object.
        :rtype:   :class:`GCEInstanceTemplate`
        """
    request = '/global/instanceTemplates'
    properties = self._create_instance_properties(name, node_size=size, source=source, image=image, disk_type=disk_type, disk_auto_delete=True, external_ip=external_ip, network=network, subnetwork=subnetwork, can_ip_forward=can_ip_forward, service_accounts=service_accounts, on_host_maintenance=on_host_maintenance, internal_ip=internal_ip, automatic_restart=automatic_restart, preemptible=preemptible, tags=tags, metadata=metadata, description=description, disks_gce_struct=disks_gce_struct, nic_gce_struct=nic_gce_struct, use_selflinks=False)
    request_data = {'name': name, 'description': description, 'properties': properties}
    self.connection.async_request(request, method='POST', data=request_data)
    return self.ex_get_instancetemplate(name)