from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
@staticmethod
def search_svc_object_by_name(service, svc_obj_name=None):
    """
        Return service object by name
        Args:
            service: Service object
            svc_obj_name: Name of service object to find

        Returns: Service object if found else None

        """
    if not svc_obj_name:
        return None
    for svc_object in service.list():
        svc_obj = service.get(svc_object)
        if svc_obj.name == svc_obj_name:
            return svc_obj
    return None