import dataclasses
import typing
from spnego._ntlm_raw.crypto import is_ntlm_hash
from spnego.exceptions import InvalidCredentialError, NoCredentialError
@property
def supported_protocols(self) -> typing.List[str]:
    """List of protocols the credential can be used for."""
    return ['kerberos']