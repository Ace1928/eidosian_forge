from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
def fail_missing(racl, fail):
    if fail and racl == []:
        _raise_error('no entries removed on the provided match_criteria')