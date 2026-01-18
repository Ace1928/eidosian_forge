from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote_plus
def getOvhClient(ansibleModule):
    endpoint = ansibleModule.params.get('endpoint')
    application_key = ansibleModule.params.get('application_key')
    application_secret = ansibleModule.params.get('application_secret')
    consumer_key = ansibleModule.params.get('consumer_key')
    return ovh.Client(endpoint=endpoint, application_key=application_key, application_secret=application_secret, consumer_key=consumer_key)