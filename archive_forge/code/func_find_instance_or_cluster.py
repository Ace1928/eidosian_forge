from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from troveclient.i18n import _
def find_instance_or_cluster(database_client_manager, instance_or_cluster):
    """Returns an instance or cluster, found by ID or name,
    along with the type of resource, instance or cluster.
    Raises CommandError if none is found.
    """
    db_instances = database_client_manager.instances
    try:
        return (osc_utils.find_resource(db_instances, instance_or_cluster), 'instance')
    except exceptions.CommandError:
        db_clusters = database_client_manager.clusters
        try:
            return (osc_utils.find_resource(db_clusters, instance_or_cluster), 'cluster')
        except exceptions.CommandError:
            raise exceptions.CommandError(_("No instance or cluster with a name or ID of '%s' exists.") % instance_or_cluster)