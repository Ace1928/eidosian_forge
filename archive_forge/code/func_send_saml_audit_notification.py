import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def send_saml_audit_notification(action, user_id, group_ids, identity_provider, protocol, token_id, outcome):
    """Send notification to inform observers about SAML events.

    :param action: Action being audited
    :type action: str
    :param user_id: User ID from Keystone token
    :type user_id: str
    :param group_ids: List of Group IDs from Keystone token
    :type group_ids: list
    :param identity_provider: ID of the IdP from the Keystone token
    :type identity_provider: str or None
    :param protocol: Protocol ID for IdP from the Keystone token
    :type protocol: str
    :param token_id: audit_id from Keystone token
    :type token_id: str or None
    :param outcome: One of :class:`pycadf.cadftaxonomy`
    :type outcome: str
    """
    initiator = build_audit_initiator()
    target = resource.Resource(typeURI=taxonomy.ACCOUNT_USER)
    audit_type = SAML_AUDIT_TYPE
    user_id = user_id or taxonomy.UNKNOWN
    token_id = token_id or taxonomy.UNKNOWN
    group_ids = group_ids or []
    cred = credential.FederatedCredential(token=token_id, type=audit_type, identity_provider=identity_provider, user=user_id, groups=group_ids)
    initiator.credential = cred
    event_type = '%s.%s' % (SERVICE, action)
    _send_audit_notification(action, initiator, outcome, target, event_type)