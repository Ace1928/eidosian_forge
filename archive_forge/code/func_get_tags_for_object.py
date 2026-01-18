from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_tags_for_object(self, tag_service=None, tag_assoc_svc=None, dobj=None, tags=None):
    """
        Return tag objects associated with an object
        Args:
            dobj: Dynamic object
            tag_service: Tag service object
            tag_assoc_svc: Tag Association object
            tags: List or set to which the tag objects are being added, reference is returned by the method
        Returns: Tag objects associated with the given object
        """
    if tags is None:
        tags = []
    if not (isinstance(tags, list) or isinstance(tags, set)):
        self.module.fail_json(msg="The parameter 'tags' must be of type 'list' or 'set', but type %s was passed" % type(tags))
    if not dobj:
        return tags
    if not tag_service:
        tag_service = self.api_client.tagging.Tag
    if not tag_assoc_svc:
        tag_assoc_svc = self.api_client.tagging.TagAssociation
    tag_ids = tag_assoc_svc.list_attached_tags(dobj)
    add_tag = tags.append if isinstance(tags, list) else tags.add
    for tag_id in tag_ids:
        add_tag(tag_service.get(tag_id))
    return tags