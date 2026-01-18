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
def test_headers_trailers():
    """Ensure that server-sent headers/trailers make it through."""

    class HeadersTrailersFlightServer(FlightServerBase):

        def get_flight_info(self, context, descriptor):
            context.add_header('x-header', 'header-value')
            context.add_header('x-header-bin', 'header\x01value')
            context.add_trailer('x-trailer', 'trailer-value')
            context.add_trailer('x-trailer-bin', 'trailer\x01value')
            return flight.FlightInfo(pa.schema([]), descriptor, [], -1, -1)

    class HeadersTrailersMiddlewareFactory(ClientMiddlewareFactory):

        def __init__(self):
            self.headers = []

        def start_call(self, info):
            return HeadersTrailersMiddleware(self)

    class HeadersTrailersMiddleware(ClientMiddleware):

        def __init__(self, factory):
            self.factory = factory

        def received_headers(self, headers):
            for key, values in headers.items():
                for value in values:
                    self.factory.headers.append((key, value))
    factory = HeadersTrailersMiddlewareFactory()
    with HeadersTrailersFlightServer() as server, FlightClient(('localhost', server.port), middleware=[factory]) as client:
        client.get_flight_info(flight.FlightDescriptor.for_path(''))
        assert ('x-header', 'header-value') in factory.headers
        assert ('x-header-bin', b'header\x01value') in factory.headers
        assert ('x-trailer', 'trailer-value') in factory.headers
        assert ('x-trailer-bin', b'trailer\x01value') in factory.headers