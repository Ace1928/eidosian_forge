import http.client
import http.server
import threading
from oslo_utils import units
def get_http_uri(test, image_id):
    uri = 'http://%(http_ip)s:%(http_port)d/images/' % {'http_ip': test.http_ip, 'http_port': test.http_port}
    uri += image_id
    return uri