import os.path
from magnumclient.common import cliutils as utils
from magnumclient.i18n import _
@utils.arg('--csr', metavar='<csr>', help=_('File path of the csr file to send to Magnum to get signed.'))
@utils.arg('--cluster', required=False, metavar='<cluster>', help=_('ID or name of the cluster.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_ca_sign(cs, args):
    """Generate the CA certificate for a cluster."""
    opts = {'cluster_uuid': _get_target_uuid(cs, args)}
    if args.csr is None or not os.path.isfile(args.csr):
        print('A CSR must be provided.')
        return
    with open(args.csr, 'r') as f:
        opts['csr'] = f.read()
    cert = cs.certificates.create(**opts)
    _show_cert(cert)