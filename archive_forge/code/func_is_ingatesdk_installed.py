from __future__ import (absolute_import, division, print_function)
def is_ingatesdk_installed(module):
    if not HAS_INGATESDK:
        module.fail_json(msg='The Ingate Python SDK module is required for this module.')