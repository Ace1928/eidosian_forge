from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def process_export(self, metadata):
    from ansible_collections.fortinet.fortimanager.plugins.module_utils.exported_schema import schemas as exported_schema_inventory
    params = self.module.params
    export_selectors = params['export_playbooks']['selector']
    export_path = './'
    if params['export_playbooks'].get('path', None):
        export_path = params['export_playbooks']['path']
    log = open('%s/export.log' % export_path, 'w')
    log.write('Export time: %s\n' % str(datetime.datetime.now()))
    for selector in export_selectors:
        if selector == 'all':
            continue
        export_meta = metadata[selector]
        export_meta_param = export_meta['params']
        export_meta_urls = export_meta['urls']
        if not params['export_playbooks']['params'] or selector not in params['export_playbooks']['params']:
            self.module.fail_json('parameter export_playbooks->params needs entry:%s' % selector)
        if not len(export_meta_urls):
            raise AssertionError('Invalid schema.')
        url_tokens = export_meta_urls[0].split('/')
        required_params = list()
        for _param in export_meta_param:
            if '{%s}' % _param == url_tokens[-1]:
                continue
            required_params.append(_param)
        for _param in required_params:
            if _param not in params['export_playbooks']['params'][selector]:
                self.module.fail_json('required parameters for selector %s: %s' % (selector, required_params))
    if 'all' in export_selectors:
        if 'all' not in params['export_playbooks']['params'] or 'adom' not in params['export_playbooks']['params']['all']:
            self.module.fail_json('required parameters for selector %s: %s' % ('all', ['adom']))
    selectors_to_process = dict()
    for selector in export_selectors:
        if selector == 'all':
            continue
        selectors_to_process[selector] = (metadata[selector], self.module.params['export_playbooks']['params'][selector])
    if 'all' in export_selectors:
        for selector in metadata:
            chosen = True
            if not len(metadata[selector]['urls']):
                raise AssertionError('Invalid Schema.')
            url_tokens = metadata[selector]['urls'][0].split('/')
            for _param in metadata[selector]['params']:
                if _param == 'adom':
                    continue
                elif '{%s}' % _param != url_tokens[-1]:
                    chosen = False
                    break
            if not chosen or selector in selectors_to_process:
                continue
            selectors_to_process[selector] = (metadata[selector], self.module.params['export_playbooks']['params']['all'])
    process_counter = 1
    number_selectors = len(selectors_to_process)
    for selector in selectors_to_process:
        self._process_export_per_selector(selector, selectors_to_process[selector][0], selectors_to_process[selector][1], log, export_path, '%s/%s' % (process_counter, number_selectors), exported_schema_inventory)
        process_counter += 1
    self.module.exit_json(number_of_selectors=number_selectors, number_of_valid_selectors=self._nr_valid_selectors, number_of_exported_playbooks=self._nr_exported_playbooks, system_infomation=self.system_status)