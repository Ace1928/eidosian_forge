import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def make_v3_endpoints(endpoints):
    for endpoint in (ep.to_dict() for ep in endpoints if ep.enabled):
        del endpoint['service_id']
        del endpoint['legacy_endpoint_id']
        del endpoint['enabled']
        endpoint['region'] = endpoint['region_id']
        try:
            formatted_url = utils.format_url(endpoint['url'], d, silent_keyerror_failures=silent_keyerror_failures)
            if formatted_url:
                endpoint['url'] = formatted_url
            else:
                continue
        except exception.MalformedEndpoint:
            continue
        yield endpoint