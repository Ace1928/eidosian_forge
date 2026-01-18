from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from magnumclient.v1 import basemodels
@utils.arg('cluster_templates', metavar='<cluster_templates>', nargs='+', help=_('ID or name of the (cluster template)s to delete.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_template_delete(cs, args):
    """Delete specified cluster template."""
    for cluster_template in args.cluster_templates:
        try:
            cs.cluster_templates.delete(cluster_template)
            print('Request to delete cluster template %s has been accepted.' % cluster_template)
        except Exception as e:
            print('Delete for cluster template %(cluster_template)s failed: %(e)s' % {'cluster_template': cluster_template, 'e': e})