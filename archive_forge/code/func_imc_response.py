from __future__ import absolute_import, division, print_function
import datetime
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves import zip_longest
from ansible.module_utils.urls import fetch_url
def imc_response(module, rawoutput, rawinput=''):
    """ Handle IMC returned data """
    xmloutput = lxml.etree.fromstring(rawoutput)
    result = cobra.data(xmloutput)
    if xmloutput.get('errorCode') and xmloutput.get('errorDescr'):
        if rawinput:
            result['input'] = rawinput
        result['output'] = rawoutput
        result['error_code'] = xmloutput.get('errorCode')
        result['error_text'] = xmloutput.get('errorDescr')
        module.fail_json(msg='Request failed: %(error_text)s' % result, **result)
    return result