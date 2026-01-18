from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def jsonify_and_parse_output(self, xml_data):
    """ convert from XML to JSON
            extract status and error fields is present
        """
    try:
        as_str = xml_data.to_string()
    except Exception as exc:
        self.module.fail_json(msg='Error running zapi in to_string: %s' % str(exc))
    try:
        as_dict = xmltodict.parse(as_str, xml_attribs=True)
    except Exception as exc:
        self.module.fail_json(msg='Error running zapi in xmltodict: %s: %s' % (as_str, str(exc)))
    try:
        as_json = json.loads(json.dumps(as_dict))
    except Exception as exc:
        self.module.fail_json(msg='Error running zapi in json load/dump: %s: %s' % (as_dict, str(exc)))
    if 'results' not in as_json:
        self.module.fail_json(msg='Error running zapi, no results field: %s: %s' % (as_str, repr(as_json)))
    errno = None
    reason = None
    response = as_json.pop('results')
    status = response.get('@status', 'no_status_attr')
    if status != 'passed':
        errno = response.get('@errno', None)
        if errno is None:
            errno = response.get('errorno', None)
        if errno is None:
            errno = 'ESTATUSFAILED'
        reason = response.get('@reason', None)
        if reason is None:
            reason = response.get('reason', None)
        if reason is None:
            reason = 'Execution failure with unknown reason.'
    for key in ('@status', '@errno', '@reason', '@xmlns'):
        try:
            del response[key]
        except KeyError:
            pass
    return (response, status, errno, reason)