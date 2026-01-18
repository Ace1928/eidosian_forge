from __future__ import annotations
from kombu.utils.compat import _detect_environment
from kombu.utils.imports import symbol_by_name
def supports_librabbitmq() -> bool | None:
    """Return true if :pypi:`librabbitmq` can be used."""
    if _detect_environment() == 'default':
        try:
            import librabbitmq
        except ImportError:
            pass
        else:
            return True
    return None