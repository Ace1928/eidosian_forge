from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def get_generic_get_iter(self, call, attribute=None, key_fields=None, query=None, attributes_list_tag='attributes-list', fail_on_error=True):
    """Method to run a generic get-iter call"""
    generic_call, error = self.call_api(call, attributes_list_tag, query, fail_on_error=fail_on_error)
    if error is not None:
        return {'error': error}
    if generic_call is None:
        return None
    if attributes_list_tag is None:
        attributes_list = generic_call
    else:
        attributes_list = generic_call.get_child_by_name(attributes_list_tag)
    if attributes_list is None:
        return None
    if key_fields is None:
        out = []
    else:
        out = {}
    iteration = 0
    for child in attributes_list.get_children():
        iteration += 1
        dic = xmltodict.parse(child.to_string(), xml_attribs=False)
        if attribute is not None:
            try:
                dic = dic[attribute]
            except KeyError as exc:
                error_message = 'Error: attribute %s not found for %s, got: %s' % (str(exc), call, dic)
                self.module.fail_json(msg=error_message, exception=traceback.format_exc())
        info = json.loads(json.dumps(dic))
        if self.translate_keys:
            info = convert_keys(info)
        if isinstance(key_fields, str):
            try:
                unique_key = _finditem(dic, key_fields)
            except KeyError as exc:
                error_message = 'Error: key %s not found for %s, got: %s' % (str(exc), call, repr(info))
                if self.error_flags['key_error']:
                    self.module.fail_json(msg=error_message, exception=traceback.format_exc())
                unique_key = 'Error_%d_key_not_found_%s' % (iteration, exc.args[0])
        elif isinstance(key_fields, tuple):
            try:
                unique_key = ':'.join([_finditem(dic, el) for el in key_fields])
            except KeyError as exc:
                error_message = 'Error: key %s not found for %s, got: %s' % (str(exc), call, repr(info))
                if self.error_flags['key_error']:
                    self.module.fail_json(msg=error_message, exception=traceback.format_exc())
                unique_key = 'Error_%d_key_not_found_%s' % (iteration, exc.args[0])
        else:
            unique_key = None
        if unique_key is not None:
            out = out.copy()
            out.update({unique_key: info})
        else:
            out.append(info)
    if attributes_list_tag is None and key_fields is None:
        if len(out) == 1:
            out = out[0]
        elif len(out) > 1:
            dic = dict()
            key_count = 0
            for item in out:
                if not isinstance(item, dict):
                    key_count = -1
                    break
                dic.update(item)
                key_count += len(item)
            if key_count == len(dic):
                out = dic
    return out