import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def quota_usage_show(quotas):
    quota_list = []
    quotas_info_dict = quotas._info
    for resource in quotas_info_dict.keys():
        good_name = False
        for name in _quota_resources:
            if resource.startswith(name):
                good_name = True
        if not good_name:
            continue
        quota_info = getattr(quotas, resource, None)
        quota_info['Type'] = resource
        quota_info = dict(((k.capitalize(), v) for k, v in quota_info.items()))
        quota_list.append(quota_info)
    print_list(quota_list, _quota_infos)