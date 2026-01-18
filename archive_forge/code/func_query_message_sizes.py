import base64
import logging
import struct
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, unify_credentials
from spnego._gss import GSSAPIProxy
from spnego._spnego import (
from spnego._sspi import SSPIProxy
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
def query_message_sizes(self) -> SecPkgContextSizes:
    if not self.complete:
        raise NoContextError(context_msg='Cannot get message sizes until context has been established')
    return self._context.query_message_sizes()