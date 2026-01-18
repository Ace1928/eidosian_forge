import logging
import posixpath
import random
import re
import http.client as httplib
import urllib.parse as urlparse
from oslo_vmware._i18n import _
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def get_recommended_datastore_clone(session, dsc_ref, clone_spec, vm_ref, folder, name, resource_pool=None, host_ref=None):
    """Returns a key which identifies the most recommended datastore from the
    specified datastore cluster where the specified VM can be cloned to.
    """
    sp_spec = vim_util.storage_placement_spec(session.vim.client.factory, dsc_ref, 'clone', clone_spec=clone_spec, vm_ref=vm_ref, folder=folder, clone_name=name, res_pool_ref=resource_pool, host_ref=host_ref)
    return get_recommended_datastore(session, sp_spec)