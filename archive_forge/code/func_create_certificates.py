import datetime
import glob
import os
from typing import Dict, Optional, Tuple, Union
import zmq
def create_certificates(key_dir: Union[str, os.PathLike], name: str, metadata: Optional[Dict[str, str]]=None) -> Tuple[str, str]:
    """Create zmq certificates.

    Returns the file paths to the public and secret certificate files.
    """
    public_key, secret_key = zmq.curve_keypair()
    base_filename = os.path.join(key_dir, name)
    secret_key_file = f'{base_filename}.key_secret'
    public_key_file = f'{base_filename}.key'
    now = datetime.datetime.now()
    _write_key_file(public_key_file, _cert_public_banner.format(now), public_key)
    _write_key_file(secret_key_file, _cert_secret_banner.format(now), public_key, secret_key=secret_key, metadata=metadata)
    return (public_key_file, secret_key_file)