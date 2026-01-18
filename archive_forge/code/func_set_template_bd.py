from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_template_bd(self, bd, fail_module=True):
    """
        Get template bridge domain item that matches the name of a bd.
        :param bd: Name of the bd to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Template bd item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['template'])
    kv_list = [KVPair('name', bd)]
    match, existing = self.get_object_from_list(self.schema_objects['template'].details.get('bds'), kv_list)
    if not match and fail_module:
        msg = "Provided BD '{0}' not matching existing bd(s): {1}".format(bd, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['template_bd'] = match