import argparse
import code
import gzip
import ssl
import sys
import threading
import time
import zlib
from urllib.parse import urlparse
import websocket
def recv_ws() -> None:
    while True:
        opcode, data = recv()
        msg = None
        if opcode == websocket.ABNF.OPCODE_TEXT and isinstance(data, bytes):
            data = str(data, 'utf-8')
        if isinstance(data, bytes) and len(data) > 2 and (data[:2] == b'\x1f\x8b'):
            try:
                data = '[gzip] ' + str(gzip.decompress(data), 'utf-8')
            except:
                pass
        elif isinstance(data, bytes):
            try:
                data = '[zlib] ' + str(zlib.decompress(data, -zlib.MAX_WBITS), 'utf-8')
            except:
                pass
        if isinstance(data, bytes):
            data = repr(data)
        if args.verbose:
            msg = f'{websocket.ABNF.OPCODE_MAP.get(opcode)}: {data}'
        else:
            msg = data
        if msg is not None:
            if args.timings:
                console.write(f'{time.time() - start_time}: {msg}')
            else:
                console.write(msg)
        if opcode == websocket.ABNF.OPCODE_CLOSE:
            break