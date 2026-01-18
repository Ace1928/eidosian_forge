import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography import x509
from cryptography.x509.oid import NameOID
from oslo_serialization import base64
from oslo_serialization import jsonutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
def handle_json_from_file(json_arg):
    """Attempts to read JSON file by the file url.

    :param json_arg: May be a file name containing the JSON.
    :returns: A list or dictionary parsed from JSON.
    """
    try:
        with open(json_arg, 'r') as f:
            json_arg = f.read().strip()
            json_arg = jsonutils.loads(json_arg)
    except IOError as e:
        err = _("Cannot get JSON from file '%(file)s'. Error: %(err)s") % {'err': e, 'file': json_arg}
        raise exc.InvalidAttribute(err)
    except ValueError as e:
        err = _("For JSON: '%(string)s', error: '%(err)s'") % {'err': e, 'string': json_arg}
        raise exc.InvalidAttribute(err)
    return json_arg