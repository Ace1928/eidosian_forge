from urllib import parse
from cinderclient import api_versions
from cinderclient import base
@api_versions.wraps('3.11')
def set_keys(self, metadata):
    """Set group specs on a group type.

        :param type : The :class:`GroupType` to set spec on
        :param metadata: A dict of key/value pairs to be set
        """
    body = {'group_specs': metadata}
    return self.manager._create('/group_types/%s/group_specs' % base.getid(self), body, 'group_specs', return_raw=True)