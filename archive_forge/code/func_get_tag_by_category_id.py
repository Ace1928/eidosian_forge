from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tag_by_category_id(self, tag_name=None, category_id=None):
    """
        Return tag object by category id
        Args:
            tag_name: Name of tag
            category_id: Id of category
        Returns: Tag object if found else None
        """
    if tag_name is None:
        return None
    if category_id is None:
        return self.search_svc_object_by_name(service=self.api_client.tagging.Tag, svc_obj_name=tag_name)
    result = None
    for tag_id in self.api_client.tagging.Tag.list_tags_for_category(category_id):
        tag_obj = self.api_client.tagging.Tag.get(tag_id)
        if tag_obj.name == tag_name:
            result = tag_obj
            break
    return result