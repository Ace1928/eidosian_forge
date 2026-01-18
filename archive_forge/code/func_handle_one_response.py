import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
def handle_one_response(self):
    start = time.time()
    headers_set = []
    headers_sent = []
    request_input = self.environ['eventlet.input']
    request_input.headers_sent = headers_sent
    wfile = self.wfile
    result = None
    use_chunked = [False]
    length = [0]
    status_code = [200]

    def write(data):
        towrite = []
        if not headers_set:
            raise AssertionError('write() before start_response()')
        elif not headers_sent:
            status, response_headers = headers_set
            headers_sent.append(1)
            header_list = [header[0].lower() for header in response_headers]
            towrite.append(('%s %s\r\n' % (self.protocol_version, status)).encode())
            for header in response_headers:
                towrite.append(('%s: %s\r\n' % header).encode('latin-1'))
            if 'date' not in header_list:
                towrite.append(('Date: %s\r\n' % (format_date_time(time.time()),)).encode())
            client_conn = self.headers.get('Connection', '').lower()
            send_keep_alive = False
            if self.close_connection == 0 and self.server.keepalive and (client_conn == 'keep-alive' or (self.request_version == 'HTTP/1.1' and (not client_conn == 'close'))):
                send_keep_alive = client_conn == 'keep-alive'
                self.close_connection = 0
            else:
                self.close_connection = 1
            if 'content-length' not in header_list:
                if self.request_version == 'HTTP/1.1':
                    use_chunked[0] = True
                    towrite.append(b'Transfer-Encoding: chunked\r\n')
                elif 'content-length' not in header_list:
                    self.close_connection = 1
            if self.close_connection:
                towrite.append(b'Connection: close\r\n')
            elif send_keep_alive:
                towrite.append(b'Connection: keep-alive\r\n')
                int_timeout = int(self.server.keepalive or 0)
                if not isinstance(self.server.keepalive, bool) and int_timeout:
                    towrite.append(b'Keep-Alive: timeout=%d\r\n' % int_timeout)
            towrite.append(b'\r\n')
        if use_chunked[0]:
            towrite.append(('%x' % (len(data),)).encode() + b'\r\n' + data + b'\r\n')
        else:
            towrite.append(data)
        wfile.writelines(towrite)
        wfile.flush()
        length[0] = length[0] + sum(map(len, towrite))

    def start_response(status, response_headers, exc_info=None):
        status_code[0] = status.split()[0]
        if exc_info:
            try:
                if headers_sent:
                    raise exc_info[1].with_traceback(exc_info[2])
            finally:
                exc_info = None
        if self.capitalize_response_headers:

            def cap(x):
                return x.encode('latin1').capitalize().decode('latin1')
            response_headers = [('-'.join([cap(x) for x in key.split('-')]), value) for key, value in response_headers]
        headers_set[:] = [status, response_headers]
        return write
    try:
        try:
            WSGI_LOCAL.already_handled = False
            result = self.application(self.environ, start_response)
            if headers_set and (not headers_sent) and hasattr(result, '__len__'):
                if 'Content-Length' not in [h for h, _v in headers_set[1]]:
                    headers_set[1].append(('Content-Length', str(sum(map(len, result)))))
                if request_input.should_send_hundred_continue:
                    self.close_connection = 1
            towrite = []
            towrite_size = 0
            just_written_size = 0
            minimum_write_chunk_size = int(self.environ.get('eventlet.minimum_write_chunk_size', self.minimum_chunk_size))
            for data in result:
                if len(data) == 0:
                    continue
                if isinstance(data, str):
                    data = data.encode('ascii')
                towrite.append(data)
                towrite_size += len(data)
                if towrite_size >= minimum_write_chunk_size:
                    write(b''.join(towrite))
                    towrite = []
                    just_written_size = towrite_size
                    towrite_size = 0
            if WSGI_LOCAL.already_handled:
                self.close_connection = 1
                return
            if towrite:
                just_written_size = towrite_size
                write(b''.join(towrite))
            if not headers_sent or (use_chunked[0] and just_written_size):
                write(b'')
        except (Exception, eventlet.Timeout):
            self.close_connection = 1
            tb = traceback.format_exc()
            self.server.log.info(tb)
            if not headers_sent:
                err_body = tb.encode() if self.server.debug else b''
                start_response('500 Internal Server Error', [('Content-type', 'text/plain'), ('Content-length', len(err_body))])
                write(err_body)
    finally:
        if hasattr(result, 'close'):
            result.close()
        if request_input.should_send_hundred_continue:
            self.close_connection = 1
        if request_input.chunked_input or request_input.position < (request_input.content_length or 0):
            if self.close_connection == 0:
                try:
                    request_input.discard()
                except ChunkReadError as e:
                    self.close_connection = 1
                    self.server.log.error(('chunked encoding error while discarding request body.' + ' client={0} request="{1}" error="{2}"').format(self.get_client_address()[0], self.requestline, e))
                except OSError as e:
                    self.close_connection = 1
                    self.server.log.error(('I/O error while discarding request body.' + ' client={0} request="{1}" error="{2}"').format(self.get_client_address()[0], self.requestline, e))
        finish = time.time()
        for hook, args, kwargs in self.environ['eventlet.posthooks']:
            hook(self.environ, *args, **kwargs)
        if self.server.log_output:
            client_host, client_port = self.get_client_address()
            self.server.log.info(self.server.log_format % {'client_ip': client_host, 'client_port': client_port, 'date_time': self.log_date_time_string(), 'request_line': self.requestline, 'status_code': status_code[0], 'body_length': length[0], 'wall_seconds': finish - start})