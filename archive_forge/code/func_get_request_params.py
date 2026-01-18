from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
def get_request_params(self) -> List[ExtensionParameter]:
    """
        Build request parameters.

        """
    return _build_parameters(self.server_no_context_takeover, self.client_no_context_takeover, self.server_max_window_bits, self.client_max_window_bits)