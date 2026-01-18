from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_template_anp(self, anp, fail_module=True):
    """
        Get template application profile item that matches the name of an anp.
        :param anp: Name of the anp to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Template anp item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['template'])
    kv_list = [KVPair('name', anp)]
    match, existing = self.get_object_from_list(self.schema_objects['template'].details.get('anps'), kv_list)
    if not match and fail_module:
        msg = "Provided ANP '{0}' not matching existing anp(s): {1}".format(anp, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['template_anp'] = match