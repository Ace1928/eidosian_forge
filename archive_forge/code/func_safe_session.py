from __future__ import annotations
import threading
from collections import ChainMap
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Iterator, cast
from streamlit.connections import BaseConnection
from streamlit.connections.util import (
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_data
@contextmanager
def safe_session(self) -> Iterator[Session]:
    """Grab the underlying Snowpark session in a thread-safe manner.

        As operations on a Snowpark session are not thread safe, we need to take care
        when using a session in the context of a Streamlit app where each script run
        occurs in its own thread. Using the contextmanager pattern to do this ensures
        that access on this connection's underlying Session is done in a thread-safe
        manner.

        Information on how to use Snowpark sessions can be found in the `Snowpark documentation
        <https://docs.snowflake.com/en/developer-guide/snowpark/python/working-with-dataframes>`_.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> conn = st.connection("snowpark")
        >>> with conn.safe_session() as session:
        ...     df = session.table("mytable").limit(10).to_pandas()
        ...
        >>> st.dataframe(df)
        """
    with self._lock:
        yield self.session