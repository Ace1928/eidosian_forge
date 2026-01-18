from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
def send_with_data_passthrough(self, event: Event) -> Optional[List[bytes]]:
    """Identical to :meth:`send`, except that in situations where
        :meth:`send` returns a single :term:`bytes-like object`, this instead
        returns a list of them -- and when sending a :class:`Data` event, this
        list is guaranteed to contain the exact object you passed in as
        :attr:`Data.data`. See :ref:`sendfile` for discussion.

        """
    if self.our_state is ERROR:
        raise LocalProtocolError("Can't send data when our state is ERROR")
    try:
        if type(event) is Response:
            event = self._clean_up_response_headers_for_sending(event)
        writer = self._writer
        self._process_event(self.our_role, event)
        if type(event) is ConnectionClosed:
            return None
        else:
            assert writer is not None
            data_list: List[bytes] = []
            writer(event, data_list.append)
            return data_list
    except:
        self._process_error(self.our_role)
        raise