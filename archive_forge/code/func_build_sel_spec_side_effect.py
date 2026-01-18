import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def build_sel_spec_side_effect(client_factory, name):
    if name == 'visitFolders':
        return sel_spec
    elif name == 'rp_to_rp':
        return rp_to_rp_sel_spec
    elif name == 'rp_to_vm':
        return rp_to_vm_sel_spec
    else:
        return None