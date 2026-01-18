from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
def process_response_params(self, params: Sequence[ExtensionParameter], accepted_extensions: Sequence[Extension]) -> PerMessageDeflate:
    """
        Process response parameters.

        Return an extension instance.

        """
    if any((other.name == self.name for other in accepted_extensions)):
        raise exceptions.NegotiationError(f'received duplicate {self.name}')
    server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits = _extract_parameters(params, is_server=False)
    if self.server_no_context_takeover:
        if not server_no_context_takeover:
            raise exceptions.NegotiationError('expected server_no_context_takeover')
    if self.client_no_context_takeover:
        if not client_no_context_takeover:
            client_no_context_takeover = True
    if self.server_max_window_bits is None:
        pass
    elif server_max_window_bits is None:
        raise exceptions.NegotiationError('expected server_max_window_bits')
    elif server_max_window_bits > self.server_max_window_bits:
        raise exceptions.NegotiationError('unsupported server_max_window_bits')
    if self.client_max_window_bits is None:
        if client_max_window_bits is not None:
            raise exceptions.NegotiationError('unexpected client_max_window_bits')
    elif self.client_max_window_bits is True:
        pass
    elif client_max_window_bits is None:
        client_max_window_bits = self.client_max_window_bits
    elif client_max_window_bits > self.client_max_window_bits:
        raise exceptions.NegotiationError('unsupported client_max_window_bits')
    return PerMessageDeflate(server_no_context_takeover, client_no_context_takeover, server_max_window_bits or 15, client_max_window_bits or 15, self.compress_settings)