import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def resolve_domain(domain):
    """Return domain id.

        Input is a dictionary with a domain identified either by a ``id`` or a
        ``name``. In the latter case system will attempt to fetch domain object
        from the backend.

        :returns: domain's id
        :rtype: str

        """
    domain_id = domain.get('id') or resource_api.get_domain_by_name(domain.get('name')).get('id')
    return domain_id