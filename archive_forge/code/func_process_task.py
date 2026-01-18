from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def process_task(self, metadata, task_type):
    params = self.module.params
    selector = params[task_type]['selector']
    param_url_id = 'params' if task_type == 'facts' else 'self'
    param_url = params[task_type][param_url_id]
    param_url = param_url if param_url else {}
    param_target = params[task_type].get('target', {})
    param_target = param_target if param_target else {}
    mkey = metadata[selector].get('mkey', None)
    if mkey and (not mkey.startswith('complex:')) and (mkey not in param_target):
        modified_mkey = _get_modified_name(mkey)
        if modified_mkey in param_target:
            param_target[mkey] = param_target[modified_mkey]
            del param_target[modified_mkey]
        else:
            self.module.fail_json(msg='Must give the primary key/value in target: %s!' % mkey)
    vrange = metadata[selector].get('v_range', None)
    matched, checking_message = self._version_matched(vrange)
    if not matched:
        self.version_check_warnings.append('selector:%s %s' % (selector, checking_message))
    param_map = {}
    for param_name in metadata[selector]['params']:
        modified_name = _get_modified_name(param_name)
        if modified_name in param_url:
            param_map[param_name] = modified_name
        elif param_name in param_url:
            param_map[param_name] = param_name
        else:
            self.module.fail_json(msg='Missing param:%s' % modified_name)
    adom_value = param_url.get('adom', None)
    target_url = self._get_target_url(adom_value, metadata[selector]['urls'])
    for param in param_map:
        token_hint = '{%s}' % param
        user_param_name = param_map[param]
        token = '%s' % param_url[user_param_name] if param_url[user_param_name] else ''
        target_url = target_url.replace(token_hint, token)
    request_type = {'clone': 'clone', 'rename': 'update', 'move': 'move', 'facts': 'get'}
    api_params = {'url': target_url}
    if task_type in ['clone', 'rename']:
        api_params['data'] = param_target
    elif task_type == 'move':
        api_params['option'] = params[task_type]['action']
        api_params['target'] = param_target
    elif task_type == 'facts':
        fact_params = params['facts']
        for key in ['filter', 'sortings', 'fields', 'option']:
            if fact_params.get(key, None):
                api_params[key] = fact_params[key]
        if fact_params.get('extra_params', None):
            for key in fact_params['extra_params']:
                api_params[key] = fact_params['extra_params'][key]
    response = self.conn.send_request(request_type[task_type], [api_params])
    self.do_exit(response, changed=task_type != 'facts')