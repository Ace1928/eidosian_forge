from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.federation import utils as federation_utils
from keystone.i18n import _
def group_membership():
    """Return a list of dictionaries serialized as strings.

        The expected return structure is::

        ['JSON:{"name":"group1","domain":{"name":"Default"}}',
        'JSON:{"name":"group2","domain":{"name":"Default"}}']
        """
    user_groups = []
    groups = PROVIDERS.identity_api.list_groups_for_user(token.user_id)
    for group in groups:
        user_group = {}
        group_domain_name = PROVIDERS.resource_api.get_domain(group['domain_id'])['name']
        user_group['name'] = group['name']
        user_group['domain'] = {'name': group_domain_name}
        user_groups.append('JSON:' + jsonutils.dumps(user_group))
    return user_groups