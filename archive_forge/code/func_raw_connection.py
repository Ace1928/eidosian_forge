from __future__ import annotations
from datetime import timedelta
from typing import TYPE_CHECKING, cast
from streamlit.connections import BaseConnection
from streamlit.connections.util import running_in_sis
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_data
@property
def raw_connection(self) -> InternalSnowflakeConnection:
    """Access the underlying Snowflake Python connector object.

        Information on how to use the Snowflake Python Connector can be found in the
        `Snowflake Python Connector documentation <https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-example>`_.
        """
    return self._instance