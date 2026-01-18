import contextlib
import decimal
import uuid
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence, Union
from sqlalchemy import (
from sqlalchemy.ext.asyncio import (
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, Session, sessionmaker
from langchain.indexes.base import RecordManager
def get_time(self) -> float:
    """Get the current server time as a timestamp.

        Please note it's critical that time is obtained from the server since
        we want a monotonic clock.
        """
    with self._make_session() as session:
        if self.dialect == 'sqlite':
            query = text("SELECT (julianday('now') - 2440587.5) * 86400.0;")
        elif self.dialect == 'postgresql':
            query = text('SELECT EXTRACT (EPOCH FROM CURRENT_TIMESTAMP);')
        else:
            raise NotImplementedError(f'Not implemented for dialect {self.dialect}')
        dt = session.execute(query).scalar()
        if isinstance(dt, decimal.Decimal):
            dt = float(dt)
        if not isinstance(dt, float):
            raise AssertionError(f'Unexpected type for datetime: {type(dt)}')
        return dt