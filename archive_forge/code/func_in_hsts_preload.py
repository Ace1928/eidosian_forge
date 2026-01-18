import functools
import os
import typing
@functools.lru_cache(maxsize=1024)
def in_hsts_preload(host: typing.AnyStr) -> bool:
    """Determines if an IDNA-encoded host is on the HSTS preload list"""
    if isinstance(host, str):
        host = host.encode('ascii')
    labels = host.lower().split(b'.')
    if labels[-1] in _GTLD_INCLUDE_SUBDOMAINS:
        return True
    with open_pkg_binary('hstspreload.bin') as f:
        for layer, label in enumerate(labels[::-1]):
            if layer > 4:
                return False
            jump_info = _JUMPTABLE[layer][_crc8(label)]
            if jump_info is None:
                return False
            f.seek(jump_info[0])
            data = bytearray(jump_info[1])
            f.readinto(data)
            for is_leaf, include_subdomains, ent_label in _iter_entries(data):
                if is_leaf:
                    if ent_label == host:
                        return True
                    if include_subdomains and host.endswith(b'.' + ent_label):
                        return True
                elif label == ent_label:
                    break
            else:
                return False
    return False