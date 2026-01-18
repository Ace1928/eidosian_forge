from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
def replicaset_status_poll(client, module):
    """
    client - MongoDB Client
    poll - Number of times to poll
    interval - interval between polling attempts
    """
    iterations = 0
    failures = 0
    poll = module.params['poll']
    interval = module.params['interval']
    status = None
    return_doc = {}
    votes = None
    config = None
    while iterations < poll:
        try:
            iterations += 1
            replicaset_document = replicaset_status(client, module)
            members = replicaset_members(replicaset_document)
            friendly_document = replicaset_friendly_document(members)
            statuses = replicaset_statuses(friendly_document, module)
            if module.params['validate'] == 'votes':
                config = replicaset_config(client)
                votes = replicaset_votes(config)
            status, msg = replicaset_good(statuses, module, votes)
            if status:
                return_doc = {'failures': failures, 'poll': poll, 'iterations': iterations, 'msg': msg, 'replicaset': friendly_document}
                break
            else:
                failures += 1
                return_doc = {'failures': failures, 'poll': poll, 'iterations': iterations, 'msg': msg, 'replicaset': friendly_document, 'failed': True}
                if iterations == poll:
                    break
                else:
                    time.sleep(interval)
        except Exception as e:
            failures += 1
            return_doc['failed'] = True
            return_doc['msg'] = str(e)
            status = False
            if iterations == poll:
                break
            else:
                time.sleep(interval)
    return_doc['failures'] = failures
    return (status, return_doc['msg'], return_doc)