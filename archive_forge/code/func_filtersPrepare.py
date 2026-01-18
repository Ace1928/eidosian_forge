from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def filtersPrepare(target, filters):
    filter_out = []
    if target == 'system':
        for system_filter in filters:
            filter_out.append(filters[system_filter])
    else:
        for common_filter in filters:
            if isinstance(filters[common_filter], dict):
                dict_filters = filters[common_filter]
                for single_filter in dict_filters:
                    filter_out.append('--filter={label}={key}={value}'.format(label=common_filter, key=single_filter, value=dict_filters[single_filter]))
            elif target == 'image' and common_filter in ('dangling_only', 'external'):
                if common_filter == 'dangling_only' and (not filters['dangling_only']):
                    filter_out.append('-a')
                if common_filter == 'external' and filters['external']:
                    filter_out.append('--external')
            else:
                filter_out.append('--filter={label}={value}'.format(label=common_filter, value=filters[common_filter]))
    return filter_out