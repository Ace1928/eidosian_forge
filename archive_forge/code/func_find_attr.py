from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
import simplejson as json
from osc_lib import exceptions
from osc_lib.i18n import _
def find_attr(self, path, value=None, attr=None, resource=None):
    """Find a resource via attribute or ID

        Most APIs return a list wrapped by a dict with the resource
        name as key.  Some APIs (Identity) return a dict when a query
        string is present and there is one return value.  Take steps to
        unwrap these bodies and return a single dict without any resource
        wrappers.

        :param string path:
            The API-specific portion of the URL path
        :param string value:
            value to search for
        :param string attr:
            attribute to use for resource search
        :param string resource:
            plural of the object resource name; defaults to path

        For example:
            n = find(netclient, 'network', 'networks', 'matrix')
        """
    if attr is None:
        attr = 'name'
    if resource is None:
        resource = path

    def getlist(kw):
        """Do list call, unwrap resource dict if present"""
        ret = self.list(path, **kw)
        if isinstance(ret, dict) and resource in ret:
            ret = ret[resource]
        return ret
    kwargs = {attr: value}
    data = getlist(kwargs)
    if isinstance(data, dict):
        return data
    if len(data) == 1:
        return data[0]
    if len(data) > 1:
        msg = _("Multiple %(resource)s exist with %(attr)s='%(value)s'")
        raise exceptions.CommandError(msg % {'resource': resource, 'attr': attr, 'value': value})
    kwargs = {'id': value}
    data = getlist(kwargs)
    if len(data) == 1:
        return data[0]
    msg = _("No %(resource)s with a %(attr)s or ID of '%(value)s' found")
    raise exceptions.CommandError(msg % {'resource': resource, 'attr': attr, 'value': value})