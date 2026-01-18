import datetime
import os
import socket
def add_port_to_grpc_server(server, address):
    import grpc
    if os.environ.get('RAY_USE_TLS', '0').lower() in ('1', 'true'):
        server_cert_chain, private_key, ca_cert = load_certs_from_env()
        credentials = grpc.ssl_server_credentials([(private_key, server_cert_chain)], root_certificates=ca_cert, require_client_auth=ca_cert is not None)
        return server.add_secure_port(address, credentials)
    else:
        return server.add_insecure_port(address)