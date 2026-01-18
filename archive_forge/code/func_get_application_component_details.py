from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_application_component_details(self, comp_uuid=None):
    """Use REST application/applications to get application components"""
    dummy, error = self.fail_if_no_uuid()
    if error is not None:
        return (dummy, error)
    if comp_uuid is None:
        comp_uuid, error = self.get_application_component_uuid()
        if error:
            return (comp_uuid, error)
    if comp_uuid is None:
        error = 'no component for application %s' % self.app_name
        return (None, error)
    api = 'application/applications/%s/components/%s' % (self.app_uuid, comp_uuid)
    return rest_generic.get_one_record(self.rest_api, api)