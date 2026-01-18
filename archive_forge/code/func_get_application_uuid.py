from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_application_uuid(self):
    """Use REST application/applications to get application uuid"""
    error = None
    if self.app_uuid is None:
        dummy, error = self._set_application_uuid()
    return (self.app_uuid, error)