import os
import re
def get_connectable_addresses(addr):
    unsupported_transports = set()
    found = False
    for transport, kv in parse_addresses(addr):
        if transport not in SUPPORTED_TRANSPORTS:
            unsupported_transports.add(transport)
        elif transport == 'unix':
            if 'abstract' in kv:
                yield ('\x00' + kv['abstract'])
                found = True
            elif 'path' in kv:
                yield kv['path']
                found = True
    if not found:
        raise RuntimeError('DBus transports ({}) not supported. Supported: {}'.format(unsupported_transports, SUPPORTED_TRANSPORTS))