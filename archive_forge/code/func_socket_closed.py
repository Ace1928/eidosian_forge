from __future__ import annotations
import errno
import select
import sys
from typing import Any, Optional, cast
def socket_closed(self, sock: Any) -> bool:
    """Return True if we know socket has been closed, False otherwise."""
    try:
        return self.select(sock, read=True)
    except (RuntimeError, KeyError):
        raise
    except ValueError:
        return True
    except Exception:
        return True