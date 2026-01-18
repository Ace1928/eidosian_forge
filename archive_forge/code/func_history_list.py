from urllib import parse
from monascaclient.common import monasca_manager
def history_list(self, **kwargs):
    """History list of alarm state."""
    url_str = self.base_url + '/state-history/'
    if 'dimensions' in kwargs:
        dimstr = self.get_dimensions_url_string(kwargs['dimensions'])
        kwargs['dimensions'] = dimstr
    if kwargs:
        url_str = url_str + '?%s' % parse.urlencode(kwargs, True)
    resp = self.client.list(url_str)
    return resp['elements'] if type(resp) is dict else resp