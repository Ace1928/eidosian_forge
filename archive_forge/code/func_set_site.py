from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_site(self, template_name, site_name, fail_module=True):
    """
        Get site item that matches the name of a site.
        :param template_name: Name of the template to match. -> Str
        :param site_name: Name of the site to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Site item. -> Item(Int, Dict) | None
        """
    if not self.schema.get('sites'):
        msg = "No sites associated with schema '{0}'. Associate the site with the schema using (M) mso_schema_site.".format(self.schema_name)
        self.mso.fail_json(msg=msg)
    kv_list = [KVPair('siteId', self.mso.lookup_site(site_name)), KVPair('templateName', template_name)]
    match, existing = self.get_object_from_list(self.schema.get('sites'), kv_list)
    if not match and fail_module:
        msg = "Provided site '{0}' not associated with template '{1}'. Site is currently associated with template(s): {2}".format(site_name, template_name, ', '.join(existing[1::2]))
        self.mso.fail_json(msg=msg)
    self.schema_objects['site'] = match