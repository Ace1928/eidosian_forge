import http.client
import http.server
import threading
from oslo_utils import units
def setup_http(test):
    server_class = http.server.HTTPServer
    remote_server = server_class(('127.0.0.1', 0), RemoteImageHandler)
    remote_ip, remote_port = remote_server.server_address

    def serve_requests(httpd):
        httpd.serve_forever()
    threading.Thread(target=serve_requests, args=(remote_server,)).start()
    test.http_server = remote_server
    test.http_ip = remote_ip
    test.http_port = remote_port
    test.addCleanup(test.http_server.shutdown)