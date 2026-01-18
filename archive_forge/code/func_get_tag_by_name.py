from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tag_by_name(self, tag_name=None):
    """
        Return tag object by name
        Args:
            tag_name: Name of tag

        Returns: Tag object if found else None
        """
    if not tag_name:
        return None
    return self.search_svc_object_by_name(service=self.api_client.tagging.Tag, svc_obj_name=tag_name)