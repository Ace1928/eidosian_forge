from __future__ import absolute_import, division, print_function
import os
import functools
from pprint import pformat
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.parsing.splitter import parse_kv, split_args
from ansible.utils.display import Display
from ansible.module_utils.six import raise_from
from importlib.metadata import version
def make_netbox_call(nb_endpoint, filters=None):
    """
    Wrapper for calls to NetBox and handle any possible errors.

    Args:
        nb_endpoint (object): The NetBox endpoint object to make calls.

    Returns:
        results (object): Pynetbox result.

    Raises:
        AnsibleError: Ansible Error containing an error message.
    """
    try:
        if filters:
            results = nb_endpoint.filter(**filters)
        else:
            results = nb_endpoint.all()
    except pynetbox.RequestError as e:
        if e.req.status_code == 404 and 'plugins' in e:
            raise AnsibleError('{0} - Not a valid plugin endpoint, please make sure to provide valid plugin endpoint.'.format(e.error))
        else:
            raise AnsibleError(e.error)
    return results