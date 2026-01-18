from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
from keystone.limit.models import base
def check_limit(self, limits):
    """Check the input limits satisfy the related project tree or not.

        1. Ensure the input is legal.
        2. Ensure the input will not break the exist limit tree.

        """
    for limit in limits:
        project_id = limit.get('project_id')
        domain_id = limit.get('domain_id')
        resource_name = limit['resource_name']
        resource_limit = limit['resource_limit']
        service_id = limit['service_id']
        region_id = limit.get('region_id')
        try:
            if project_id:
                parent_id = PROVIDERS.resource_api.get_project(project_id)['parent_id']
                parent_limit = list(filter(lambda x: x.get('domain_id') == parent_id and x['service_id'] == service_id and (x.get('region_id') == region_id) and (x['resource_name'] == resource_name), limits))
                if parent_limit:
                    if resource_limit > parent_limit[0]['resource_limit']:
                        error = _('The value of the limit which project is %(project_id)s should not bigger than its parent domain %(domain_id)s.') % {'project_id': project_id, 'domain_id': parent_limit[0]['domain_id']}
                        raise exception.InvalidLimit(reason=error)
                    continue
            else:
                parent_id = None
            self._check_limit(resource_name, service_id, region_id, resource_limit, domain_id=domain_id, parent_id=parent_id)
        except exception.InvalidLimit:
            error = "The resource limit (%(level)s: %(id)s, resource_name: %(resource_name)s, resource_limit: %(resource_limit)s, service_id: %(service_id)s, region_id: %(region_id)s) doesn't satisfy current hierarchy model." % {'level': 'project_id' if project_id else 'domain_id', 'id': project_id or domain_id, 'resource_name': resource_name, 'resource_limit': resource_limit, 'service_id': service_id, 'region_id': region_id}
            tr_error = _("The resource limit (%(level)s: %(id)s, resource_name: %(resource_name)s, resource_limit: %(resource_limit)s, service_id: %(service_id)s, region_id: %(region_id)s) doesn't satisfy current hierarchy model.") % {'level': 'project_id' if project_id else 'domain_id', 'id': project_id or domain_id, 'resource_name': resource_name, 'resource_limit': resource_limit, 'service_id': service_id, 'region_id': region_id}
            LOG.error(error)
            raise exception.InvalidLimit(reason=tr_error)