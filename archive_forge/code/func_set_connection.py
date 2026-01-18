from typing import Any, Dict, List, Optional
from langchain_community.graphs.graph_store import GraphStore
def set_connection(self, conn: Any) -> None:
    from pyTigerGraph import TigerGraphConnection
    if not isinstance(conn, TigerGraphConnection):
        msg = '**conn** parameter must inherit from TigerGraphConnection'
        raise TypeError(msg)
    if conn.ai.nlqs_host is None:
        msg = '**conn** parameter does not have nlqs_host parameter defined.\n                     Define hostname of NLQS service.'
        raise ConnectionError(msg)
    self._conn: TigerGraphConnection = conn
    self.set_schema()