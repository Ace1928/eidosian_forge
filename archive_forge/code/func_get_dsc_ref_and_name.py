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
def get_dsc_ref_and_name(session, dsc_val):
    """Return reference and name of the specified datastore cluster.

    :param ds_val: datastore cluster name or datastore cluster moid
    :return: tuple of dastastore cluster moref and datastore cluster name
    """
    if re.match('group-p\\d+', dsc_val):
        dsc_ref = vim_util.get_moref(dsc_val, 'StoragePod')
        try:
            dsc_name = session.invoke_api(vim_util, 'get_object_property', session.vim, dsc_ref, 'name')
            return (dsc_ref, dsc_name)
        except exceptions.ManagedObjectNotFoundException:
            pass
    result = session.invoke_api(vim_util, 'get_objects', session.vim, 'StoragePod', 100, ['name'])
    with vim_util.WithRetrieval(session.vim, result) as objs:
        for obj in objs:
            if not hasattr(obj, 'propSet'):
                continue
            if obj.propSet[0].val == dsc_val:
                return (obj.obj, dsc_val)
    return (None, None)