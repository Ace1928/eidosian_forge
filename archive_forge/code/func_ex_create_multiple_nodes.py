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
def ex_create_multiple_nodes(self, base_name, size, image, number, location=None, ex_network='default', ex_subnetwork=None, ex_tags=None, ex_metadata=None, ignore_errors=True, use_existing_disk=True, poll_interval=2, external_ip='ephemeral', internal_ip=None, ex_disk_type='pd-standard', ex_disk_auto_delete=True, ex_service_accounts=None, timeout=DEFAULT_TASK_COMPLETION_TIMEOUT, description=None, ex_can_ip_forward=None, ex_disks_gce_struct=None, ex_nic_gce_struct=None, ex_on_host_maintenance=None, ex_automatic_restart=None, ex_image_family=None, ex_preemptible=None, ex_labels=None, ex_disk_size=None):
    """
        Create multiple nodes and return a list of Node objects.

        Nodes will be named with the base name and a number.  For example, if
        the base name is 'libcloud' and you create 3 nodes, they will be
        named::

            libcloud-000
            libcloud-001
            libcloud-002

        :param  base_name: The base name of the nodes to create.
        :type   base_name: ``str``

        :param  size: The machine type to use.
        :type   size: ``str`` or :class:`GCENodeSize`

        :param  image: The image to use to create the nodes.
        :type   image: ``str`` or :class:`GCENodeImage`

        :param  number: The number of nodes to create.
        :type   number: ``int``

        :keyword  location: The location (zone) to create the nodes in.
        :type     location: ``str`` or :class:`NodeLocation` or
                            :class:`GCEZone` or ``None``

        :keyword  ex_network: The network to associate with the nodes.
        :type     ex_network: ``str`` or :class:`GCENetwork`

        :keyword  ex_tags: A list of tags to associate with the nodes.
        :type     ex_tags: ``list`` of ``str`` or ``None``

        :keyword  ex_metadata: Metadata dictionary for instances.
        :type     ex_metadata: ``dict`` or ``None``

        :keyword  ignore_errors: If True, don't raise Exceptions if one or
                                 more nodes fails.
        :type     ignore_errors: ``bool``

        :keyword  use_existing_disk: If True and if an existing disk with the
                                     same name/location is found, use that
                                     disk instead of creating a new one.
        :type     use_existing_disk: ``bool``

        :keyword  poll_interval: Number of seconds between status checks.
        :type     poll_interval: ``int``

        :keyword  external_ip: The external IP address to use.  If 'ephemeral'
                               (default), a new non-static address will be
                               used. If 'None', then no external address will
                               be used. (Static addresses are not supported for
                               multiple node creation.)
        :type     external_ip: ``str`` or None


        :keyword  internal_ip: The private IP address to use.
        :type     internal_ip: :class:`GCEAddress` or ``str`` or ``None``

        :keyword  ex_disk_type: Specify a pd-standard (default) disk or pd-ssd
                                for an SSD disk.
        :type     ex_disk_type: ``str`` or :class:`GCEDiskType`

        :keyword  ex_disk_auto_delete: Indicate that the boot disk should be
                                       deleted when the Node is deleted. Set to
                                       True by default.
        :type     ex_disk_auto_delete: ``bool``

        :keyword  ex_service_accounts: Specify a list of serviceAccounts when
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
        :type     ex_service_accounts: ``list``

        :keyword  timeout: The number of seconds to wait for all nodes to be
                           created before timing out.
        :type     timeout: ``int``

        :keyword  description: The description of the node (instance).
        :type     description: ``str`` or ``None``

        :keyword  ex_can_ip_forward: Set to ``True`` to allow this node to
                                     send/receive non-matching src/dst packets.
        :type     ex_can_ip_forward: ``bool`` or ``None``

        :keyword  ex_preemptible: Defines whether the instance is preemptible.
                                  (If not supplied, the instance will
                                  not be preemptible)
        :type     ex_preemptible: ``bool`` or ``None``

        :keyword  ex_disks_gce_struct: Support for passing in the GCE-specific
                                       formatted disks[] structure. No attempt
                                       is made to ensure proper formatting of
                                       the disks[] structure. Using this
                                       structure obviates the need of using
                                       other disk params like 'ex_boot_disk',
                                       etc. See the GCE docs for specific
                                       details.
        :type     ex_disks_gce_struct: ``list`` or ``None``

        :keyword  ex_nic_gce_struct: Support passing in the GCE-specific
                                     formatted networkInterfaces[] structure.
                                     No attempt is made to ensure proper
                                     formatting of the networkInterfaces[]
                                     data. Using this structure obviates the
                                     need of using 'external_ip' and
                                     'ex_network'.  See the GCE docs for
                                     details.
        :type     ex_nic_gce_struct: ``list`` or ``None``

        :keyword  ex_on_host_maintenance: Defines whether node should be
                                          terminated or migrated when host
                                          machine goes down. Acceptable values
                                          are: 'MIGRATE' or 'TERMINATE' (If
                                          not supplied, value will be reset to
                                          GCE default value for the instance
                                          type.)
        :type     ex_on_host_maintenance: ``str`` or ``None``

        :keyword  ex_automatic_restart: Defines whether the instance should be
                                        automatically restarted when it is
                                        terminated by Compute Engine. (If not
                                        supplied, value will be set to the GCE
                                        default value for the instance type.)
        :type     ex_automatic_restart: ``bool`` or ``None``

        :keyword  ex_image_family: Determine image from an 'Image Family'
                                   instead of by name. 'image' should be None
                                   to use this keyword.
        :type     ex_image_family: ``str`` or ``None``

        :param    ex_labels: Label dict for node.
        :type     ex_labels: ``dict``

        :keyword  ex_disk_size: Defines size of the boot disk.
                                Integer in gigabytes.
        :type     ex_disk_size: ``int`` or ``None``

        :return:  A list of Node objects for the new nodes.
        :rtype:   ``list`` of :class:`Node`

        """
    if image and ex_disks_gce_struct:
        raise ValueError("Cannot specify both 'image' and 'ex_disks_gce_struct'.")
    if image and ex_image_family:
        raise ValueError("Cannot specify both 'image' and 'ex_image_family'")
    location = location or self.zone
    if not hasattr(location, 'name'):
        location = self.ex_get_zone(location)
    if not hasattr(size, 'name'):
        size = self.ex_get_size(size, location)
    if not hasattr(ex_network, 'name'):
        ex_network = self.ex_get_network(ex_network)
    if ex_subnetwork and (not hasattr(ex_subnetwork, 'name')):
        ex_subnetwork = self.ex_get_subnetwork(ex_subnetwork, region=self._get_region_from_zone(location))
    if ex_image_family:
        image = self.ex_get_image_from_family(ex_image_family)
    if image and (not hasattr(image, 'name')):
        image = self.ex_get_image(image)
    if not hasattr(ex_disk_type, 'name'):
        ex_disk_type = self.ex_get_disktype(ex_disk_type, zone=location)
    node_attrs = {'size': size, 'image': image, 'location': location, 'network': ex_network, 'subnetwork': ex_subnetwork, 'tags': ex_tags, 'metadata': ex_metadata, 'ignore_errors': ignore_errors, 'use_existing_disk': use_existing_disk, 'external_ip': external_ip, 'internal_ip': internal_ip, 'ex_disk_type': ex_disk_type, 'ex_disk_auto_delete': ex_disk_auto_delete, 'ex_service_accounts': ex_service_accounts, 'description': description, 'ex_can_ip_forward': ex_can_ip_forward, 'ex_disks_gce_struct': ex_disks_gce_struct, 'ex_nic_gce_struct': ex_nic_gce_struct, 'ex_on_host_maintenance': ex_on_host_maintenance, 'ex_automatic_restart': ex_automatic_restart, 'ex_preemptible': ex_preemptible, 'ex_labels': ex_labels, 'ex_disk_size': ex_disk_size}
    status_list = []
    for i in range(number):
        name = '%s-%03d' % (base_name, i)
        status = {'name': name, 'node_response': None, 'node': None}
        status_list.append(status)
    start_time = time.time()
    complete = False
    while not complete:
        if time.time() - start_time >= timeout:
            raise Exception('Timeout (%s sec) while waiting for multiple instances')
        complete = True
        time.sleep(poll_interval)
        for status in status_list:
            if not status['node']:
                if not status['node_response']:
                    self._multi_create_node(status, node_attrs)
                else:
                    self._multi_check_node(status, node_attrs)
            if not status['node']:
                complete = False
    node_list = []
    for status in status_list:
        node_list.append(status['node'])
    return node_list