from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
def enable_server_permessage_deflate(extensions: Optional[Sequence[ServerExtensionFactory]]) -> Sequence[ServerExtensionFactory]:
    """
    Enable Per-Message Deflate with default settings in server extensions.

    If the extension is already present, perhaps with non-default settings,
    the configuration isn't changed.

    """
    if extensions is None:
        extensions = []
    if not any((ext_factory.name == ServerPerMessageDeflateFactory.name for ext_factory in extensions)):
        extensions = list(extensions) + [ServerPerMessageDeflateFactory(server_max_window_bits=12, client_max_window_bits=12, compress_settings={'memLevel': 5})]
    return extensions