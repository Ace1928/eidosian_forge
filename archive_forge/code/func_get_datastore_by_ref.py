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
def get_datastore_by_ref(session, ds_ref):
    """Returns a datastore object for a given reference.

    :param session: a vmware api session object
    :param ds_ref: managed object reference of a datastore
    :rtype: a datastore object
    """
    lst_properties = ['summary.type', 'summary.name', 'summary.capacity', 'summary.freeSpace', 'summary.uncommitted']
    props = session.invoke_api(vim_util, 'get_object_properties_dict', session.vim, ds_ref, lst_properties)
    return Datastore(ds_ref, props['summary.name'], capacity=props.get('summary.capacity'), freespace=props.get('summary.freeSpace'), uncommitted=props.get('summary.uncommitted'), type=props.get('summary.type'))