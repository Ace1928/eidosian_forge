from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def adjust_list_api_tags(parent_input, parent_cur):
    iv = parent_input.get('tags')
    if not (iv and isinstance(iv, list)):
        return
    cv = parent_cur.get('tags')
    if not (cv and isinstance(cv, list)):
        return
    result = []
    for iiv in iv:
        if iiv not in cv:
            break
        result.append(iiv)
        j = cv.index(iiv)
        cv[j] = cv[-1]
        cv.pop()
    if cv:
        result.extend(cv)
    parent_cur['tags'] = result