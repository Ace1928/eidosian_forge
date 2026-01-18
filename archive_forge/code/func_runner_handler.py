import argparse
import json
import logging
import socket
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from ray.autoscaler._private.local.node_provider import LocalNodeProvider
def runner_handler(node_provider):

    class Handler(SimpleHTTPRequestHandler):
        """A custom handler for OnPremCoordinatorServer.

        Handles all requests and responses coming into and from the
        remote CoordinatorSenderNodeProvider.
        """

        def _do_header(self, response_code=200, headers=None):
            """Sends the header portion of the HTTP response.

            Args:
                response_code: Standard HTTP response code
                headers (list[tuples]): Standard HTTP response headers
            """
            if headers is None:
                headers = [('Content-type', 'application/json')]
            self.send_response(response_code)
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()

        def do_HEAD(self):
            """HTTP HEAD handler method."""
            self._do_header()

        def do_GET(self):
            """Processes requests from remote CoordinatorSenderNodeProvider."""
            if self.headers['content-length']:
                raw_data = self.rfile.read(int(self.headers['content-length'])).decode('utf-8')
                logger.info('OnPremCoordinatorServer received request: ' + str(raw_data))
                request = json.loads(raw_data)
                response = getattr(node_provider, request['type'])(*request['args'])
                logger.info('OnPremCoordinatorServer response content: ' + str(raw_data))
                response_code = 200
                message = json.dumps(response)
                self._do_header(response_code=response_code)
                self.wfile.write(message.encode())
    return Handler