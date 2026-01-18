import logging
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient.v2 import agents
from novaclient.v2 import aggregates
from novaclient.v2 import assisted_volume_snapshots
from novaclient.v2 import availability_zones
from novaclient.v2 import flavor_access
from novaclient.v2 import flavors
from novaclient.v2 import hypervisors
from novaclient.v2 import images
from novaclient.v2 import instance_action
from novaclient.v2 import instance_usage_audit_log
from novaclient.v2 import keypairs
from novaclient.v2 import limits
from novaclient.v2 import migrations
from novaclient.v2 import networks
from novaclient.v2 import quota_classes
from novaclient.v2 import quotas
from novaclient.v2 import server_external_events
from novaclient.v2 import server_groups
from novaclient.v2 import server_migrations
from novaclient.v2 import servers
from novaclient.v2 import services
from novaclient.v2 import usage
from novaclient.v2 import versions
from novaclient.v2 import volumes
def reset_timings(self):
    self.client.reset_timings()