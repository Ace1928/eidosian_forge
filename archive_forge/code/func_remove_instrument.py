import logging
import types
from typing import Any, Callable, Dict, Sequence, TypeVar
from .._abc import Instrument
@_public
def remove_instrument(self, instrument: Instrument) -> None:
    """Stop instrumenting the current run loop with the given instrument.

        Args:
          instrument (trio.abc.Instrument): The instrument to de-activate.

        Raises:
          KeyError: if the instrument is not currently active. This could
              occur either because you never added it, or because you added it
              and then it raised an unhandled exception and was automatically
              deactivated.

        """
    self['_all'].pop(instrument)
    for hookname, instruments in list(self.items()):
        if instrument in instruments:
            del instruments[instrument]
            if not instruments:
                del self[hookname]