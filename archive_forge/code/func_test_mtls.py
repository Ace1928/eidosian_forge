import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util
@pytest.mark.requires_testing_data
def test_mtls():
    """Test mutual TLS (mTLS) with gRPC."""
    certs = example_tls_certs()
    table = simple_ints_table()
    with ConstantFlightServer(tls_certificates=[certs['certificates'][0]], verify_client=True, root_certificates=certs['root_cert']) as s, FlightClient(('localhost', s.port), tls_root_certs=certs['root_cert'], cert_chain=certs['certificates'][0].cert, private_key=certs['certificates'][0].key) as client:
        data = client.do_get(flight.Ticket(b'ints')).read_all()
        assert data.equals(table)