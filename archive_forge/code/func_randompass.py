from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
def randompass():
    """
    Generate a long random password that comply to Linode requirements
    """
    import random
    import string
    random.seed()
    lower = ''.join((random.choice(string.ascii_lowercase) for x in range(6)))
    upper = ''.join((random.choice(string.ascii_uppercase) for x in range(6)))
    number = ''.join((random.choice(string.digits) for x in range(6)))
    punct = ''.join((random.choice(string.punctuation) for x in range(6)))
    p = lower + upper + number + punct
    return ''.join(random.sample(p, len(p)))