import base64
from rsa._compat import is_bytes, range
def save_pem(contents, pem_marker):
    """Saves a PEM file.

    :param contents: the contents to encode in PEM format
    :param pem_marker: the marker of the PEM content, such as 'RSA PRIVATE KEY'
        when your file has '-----BEGIN RSA PRIVATE KEY-----' and
        '-----END RSA PRIVATE KEY-----' markers.

    :return: the base64-encoded content between the start and end markers, as bytes.

    """
    pem_start, pem_end = _markers(pem_marker)
    b64 = base64.standard_b64encode(contents).replace(b'\n', b'')
    pem_lines = [pem_start]
    for block_start in range(0, len(b64), 64):
        block = b64[block_start:block_start + 64]
        pem_lines.append(block)
    pem_lines.append(pem_end)
    pem_lines.append(b'')
    return b'\n'.join(pem_lines)