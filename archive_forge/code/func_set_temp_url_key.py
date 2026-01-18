from openstack.object_store.v1 import _base
from openstack import resource
def set_temp_url_key(self, proxy, key, secondary=False):
    """Set the temporary url key for the account.

        :param proxy: The proxy to use for making this request.
        :type proxy: :class:`~openstack.proxy.Proxy`
        :param key:
          Text of the key to use.
        :param bool secondary:
          Whether this should set the secondary key. (defaults to False)
        """
    header = 'Temp-URL-Key'
    if secondary:
        header += '-2'
    return self.set_metadata(proxy, {header: key})