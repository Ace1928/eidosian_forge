import copy
import uuid
from swiftclient import client as sc
from swiftclient import utils as swiftclient_utils
from urllib import parse as urlparse
from heatclient._i18n import _
from heatclient import exc
from heatclient.v1 import software_configs
def build_signal_id(hc, args):
    if args.signal_transport != 'TEMP_URL_SIGNAL':
        return
    if getattr(args, 'os_no_client_auth', False):
        raise exc.CommandError(_('Cannot use --os-no-client-auth, auth required to create a Swift TempURL.'))
    swift_client = create_swift_client(hc.http_client.auth, hc.http_client.session, args)
    return create_temp_url(swift_client, args.name, args.timeout)