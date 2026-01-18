import asyncio
import logging
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
import pytest
import zmq
import zmq.asyncio
import zmq.auth
from zmq.tests import SkipTest, skip_pypy
def load_certs(secret_keys_dir):
    """Return server and client certificate keys"""
    server_secret_file = os.path.join(secret_keys_dir, 'server.key_secret')
    client_secret_file = os.path.join(secret_keys_dir, 'client.key_secret')
    server_public, server_secret = zmq.auth.load_certificate(server_secret_file)
    client_public, client_secret = zmq.auth.load_certificate(client_secret_file)
    return (server_public, server_secret, client_public, client_secret)