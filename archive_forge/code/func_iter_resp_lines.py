import json
import pydoc
from kubernetes import client
def iter_resp_lines(resp):
    prev = ''
    for seg in resp.read_chunked(decode_content=False):
        if isinstance(seg, bytes):
            seg = seg.decode('utf8')
        seg = prev + seg
        lines = seg.split('\n')
        if not seg.endswith('\n'):
            prev = lines[-1]
            lines = lines[:-1]
        else:
            prev = ''
        for line in lines:
            if line:
                yield line