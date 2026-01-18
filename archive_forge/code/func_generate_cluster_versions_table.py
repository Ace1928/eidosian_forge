from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import locations as api_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
def generate_cluster_versions_table(cluster_ref, platform, items):
    """Generates a table of user's cluster versions and then adds a "*" to the version that can be upgraded.

  Args:
    cluster_ref: A resource object, the cluster resource information.
    platform: A string, the platform the component is on {AWS,Azure}.
    items: A generator, an iterator (generator), of cluster versions that need
      to be looked at, to see if they are in end of life.

  Returns:
    A table with cluster information (with annotations on whether the cluster
    can be upgraded), an end of life flag used to tell whether we need to add
    any additional hints.
  """
    cluster_info_table = []
    end_of_life_flag = False
    valid_versions = _load_valid_versions(platform, cluster_ref)
    for x in items:
        if _is_end_of_life(valid_versions, x.controlPlane.version):
            x.controlPlane.version += ' *'
            end_of_life_flag = True
        cluster_info_table.append(x)
    return (iter(cluster_info_table), end_of_life_flag)