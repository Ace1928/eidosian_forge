from __future__ import absolute_import, division, print_function
import base64
import time
import os
import traceback
import sys
import hashlib
from datetime import datetime
from tempfile import NamedTemporaryFile
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.hashes import (
from ansible_collections.kubernetes.core.plugins.module_utils.selector import (
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems, string_types
from ansible.module_utils._text import to_native, to_bytes, to_text
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.urls import Request
def kubernetes_facts(self, kind, api_version, name=None, namespace=None, label_selectors=None, field_selectors=None, wait=False, wait_sleep=5, wait_timeout=120, state='present', condition=None):
    resource = self.find_resource(kind, api_version)
    api_found = bool(resource)
    if not api_found:
        return dict(resources=[], msg='Failed to find API for resource with apiVersion "{0}" and kind "{1}"'.format(api_version, kind), api_found=False)
    if not label_selectors:
        label_selectors = []
    if not field_selectors:
        field_selectors = []
    result = None
    params = dict(name=name, namespace=namespace, label_selector=','.join(label_selectors), field_selector=','.join(field_selectors))
    try:
        result = resource.get(**params)
    except BadRequestError:
        return dict(resources=[], api_found=True)
    except NotFoundError:
        if not wait or name is None:
            return dict(resources=[], api_found=True)
    except Exception as e:
        if not wait or name is None:
            err = "Exception '{0}' raised while trying to get resource using {1}".format(e, params)
            return dict(resources=[], msg=err, api_found=True)
    if not wait:
        result = result.to_dict()
        if 'items' in result:
            return dict(resources=result['items'], api_found=True)
        return dict(resources=[result], api_found=True)
    start = datetime.now()

    def _elapsed():
        return (datetime.now() - start).seconds

    def result_empty(result):
        return result is None or (result.kind.endswith('List') and (not result.get('items')))
    last_exception = None
    while result_empty(result) and _elapsed() < wait_timeout:
        try:
            result = resource.get(**params)
        except NotFoundError:
            pass
        except Exception as e:
            last_exception = e
        if not result_empty(result):
            break
        time.sleep(wait_sleep)
    if result_empty(result):
        res = dict(resources=[], api_found=True)
        if last_exception is not None:
            res['msg'] = "Exception '%s' raised while trying to get resource using %s" % (last_exception, params)
        return res
    if isinstance(result, ResourceInstance):
        satisfied_by = []
        resource_list = result.get('items', [])
        if not resource_list:
            resource_list = [result]
        for resource_instance in resource_list:
            success, res, duration = self.wait(resource, resource_instance, sleep=wait_sleep, timeout=wait_timeout, state=state, condition=condition)
            if not success:
                self.fail(msg='Failed to gather information about %s(s) even after waiting for %s seconds' % (res.get('kind'), duration))
            satisfied_by.append(res)
        return dict(resources=satisfied_by, api_found=True)
    result = result.to_dict()
    if 'items' in result:
        return dict(resources=result['items'], api_found=True)
    return dict(resources=[result], api_found=True)