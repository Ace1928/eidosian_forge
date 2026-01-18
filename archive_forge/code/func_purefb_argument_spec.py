from __future__ import absolute_import, division, print_function
from os import environ
import platform
def purefb_argument_spec():
    """Return standard base dictionary used for the argument_spec argument in AnsibleModule"""
    return dict(fb_url=dict(), api_token=dict(no_log=True))