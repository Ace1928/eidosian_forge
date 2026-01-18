from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
def process_request_params(self, params: Sequence[ExtensionParameter], accepted_extensions: Sequence[Extension]) -> Tuple[List[ExtensionParameter], PerMessageDeflate]:
    """
        Process request parameters.

        Return response params and an extension instance.

        """
    if any((other.name == self.name for other in accepted_extensions)):
        raise exceptions.NegotiationError(f'skipped duplicate {self.name}')
    server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits = _extract_parameters(params, is_server=True)
    if self.server_no_context_takeover:
        if not server_no_context_takeover:
            server_no_context_takeover = True
    if self.client_no_context_takeover:
        if not client_no_context_takeover:
            client_no_context_takeover = True
    if self.server_max_window_bits is None:
        pass
    elif server_max_window_bits is None:
        server_max_window_bits = self.server_max_window_bits
    elif server_max_window_bits > self.server_max_window_bits:
        server_max_window_bits = self.server_max_window_bits
    if self.client_max_window_bits is None:
        if client_max_window_bits is True:
            client_max_window_bits = self.client_max_window_bits
    elif client_max_window_bits is None:
        if self.require_client_max_window_bits:
            raise exceptions.NegotiationError('required client_max_window_bits')
    elif client_max_window_bits is True:
        client_max_window_bits = self.client_max_window_bits
    elif self.client_max_window_bits < client_max_window_bits:
        client_max_window_bits = self.client_max_window_bits
    return (_build_parameters(server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits), PerMessageDeflate(client_no_context_takeover, server_no_context_takeover, client_max_window_bits or 15, server_max_window_bits or 15, self.compress_settings))