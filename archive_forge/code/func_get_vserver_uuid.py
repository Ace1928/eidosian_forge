from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils.rest_generic import get_one_record
def get_vserver_uuid(rest_api, name, module=None, error_on_none=False):
    """ returns a tuple (uuid, error)
        when module is set and an error is found, fails the module and exit
        when error_on_none IS SET, force an error if vserver is not found
    """
    record, error = get_vserver(rest_api, name, 'uuid')
    if error and module:
        module.fail_json(msg='Error fetching vserver %s: %s' % (name, error))
    if not error and record is None and error_on_none:
        error = 'vserver %s does not exist or is not a data vserver.' % name
        if module:
            module.fail_json(msg='Error %s' % error)
    return (record['uuid'] if not error and record else None, error)