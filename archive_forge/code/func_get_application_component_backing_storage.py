from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_application_component_backing_storage(self):
    """Use REST application/applications to get component uuid.

           Assume a single component per application
        """
    dummy, error = self.fail_if_no_uuid()
    if error is not None:
        return (dummy, error)
    response, error = self.get_application_component_details()
    if error or response is None:
        return (response, error)
    return (response['backing_storage'], None)