import contextlib
import logging
import random
from socket import IPPROTO_TCP
from socket import TCP_NODELAY
from socket import SHUT_WR
from socket import timeout as SocketTimeout
import ssl
from os_ken import cfg
from os_ken.lib import hub
from os_ken.lib.hub import StreamServer
import os_ken.base.app_manager
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import nx_match
from os_ken.controller import ofp_event
from os_ken.controller.handler import HANDSHAKE_DISPATCHER, DEAD_DISPATCHER
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib import ip
def server_loop(self, ofp_tcp_listen_port, ofp_ssl_listen_port):
    if CONF.ctl_privkey is not None and CONF.ctl_cert is not None:
        p = 'PROTOCOL_TLS'
        ssl_args = {'ssl_ctx': ssl.SSLContext(getattr(ssl, p))}
        ssl_args['ssl_ctx'].options |= ssl.OP_NO_SSLv3 | ssl.OP_NO_SSLv2
        if CONF.ciphers is not None:
            ssl_args['ciphers'] = CONF.ciphers
        if CONF.ca_certs is not None:
            server = StreamServer((CONF.ofp_listen_host, ofp_ssl_listen_port), datapath_connection_factory, keyfile=CONF.ctl_privkey, certfile=CONF.ctl_cert, cert_reqs=ssl.CERT_REQUIRED, ca_certs=CONF.ca_certs, **ssl_args)
        else:
            server = StreamServer((CONF.ofp_listen_host, ofp_ssl_listen_port), datapath_connection_factory, keyfile=CONF.ctl_privkey, certfile=CONF.ctl_cert, **ssl_args)
    else:
        server = StreamServer((CONF.ofp_listen_host, ofp_tcp_listen_port), datapath_connection_factory)
    server.serve_forever()