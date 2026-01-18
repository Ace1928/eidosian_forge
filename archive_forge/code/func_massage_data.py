from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
def massage_data(have_or_want):
    data = deepcopy(have_or_want)
    massaged = {}
    massaged['destination_groups'] = {}
    massaged['sensor_groups'] = {}
    massaged['subscriptions'] = {}
    from pprint import pprint
    for subgroup in ['destination_groups', 'sensor_groups', 'subscriptions']:
        for item in data.get(subgroup, []):
            id = str(item.get('id'))
            if id not in massaged[subgroup].keys():
                massaged[subgroup][id] = []
            item.pop('id')
            if not item:
                item = None
            else:
                if item.get('destination'):
                    if item.get('destination').get('port'):
                        item['destination']['port'] = str(item['destination']['port'])
                    if item.get('destination').get('protocol'):
                        item['destination']['protocol'] = item['destination']['protocol'].lower()
                    if item.get('destination').get('encoding'):
                        item['destination']['encoding'] = item['destination']['encoding'].lower()
                if item.get('path'):
                    for key in ['filter_condition', 'query_condition', 'depth']:
                        if item.get('path').get(key) == 'None':
                            del item['path'][key]
                    if item.get('path').get('depth') is not None:
                        item['path']['depth'] = str(item['path']['depth'])
                if item.get('destination_group'):
                    item['destination_group'] = str(item['destination_group'])
                if item.get('sensor_group'):
                    if item.get('sensor_group').get('id'):
                        item['sensor_group']['id'] = str(item['sensor_group']['id'])
                    if item.get('sensor_group').get('sample_interval'):
                        item['sensor_group']['sample_interval'] = str(item['sensor_group']['sample_interval'])
                if item.get('destination_group') and item.get('sensor_group'):
                    item_copy = deepcopy(item)
                    del item_copy['sensor_group']
                    del item['destination_group']
                    massaged[subgroup][id].append(item_copy)
                    massaged[subgroup][id].append(item)
                    continue
                if item.get('path') and item.get('data_source'):
                    item_copy = deepcopy(item)
                    del item_copy['data_source']
                    del item['path']
                    massaged[subgroup][id].append(item_copy)
                    massaged[subgroup][id].append(item)
                    continue
            massaged[subgroup][id].append(item)
    return massaged