from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_application_details(self, template=None):
    """Use REST application/applications to get application details"""
    uuid, error = self.get_application_uuid()
    if error:
        return (uuid, error)
    if uuid is None:
        return (None, None)
    query = dict(fields='name,%s,statistics' % template) if template else None
    api = 'application/applications/%s' % uuid
    return rest_generic.get_one_record(self.rest_api, api, query)