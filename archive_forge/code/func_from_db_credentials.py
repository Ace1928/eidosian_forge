import os
from math import ceil
from typing import Any, Dict, List, Optional
@classmethod
def from_db_credentials(cls, url: Optional[str]=None, dbname: Optional[str]=None, username: Optional[str]=None, password: Optional[str]=None) -> Any:
    """Convenience constructor that builds Arango DB from credentials.

        Args:
            url: Arango DB url. Can be passed in as named arg or set as environment
                var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
            dbname: Arango DB name. Can be passed in as named arg or set as
                environment var ``ARANGODB_DBNAME``. Defaults to "_system".
            username: Can be passed in as named arg or set as environment var
                ``ARANGODB_USERNAME``. Defaults to "root".
            password: Can be passed ni as named arg or set as environment var
                ``ARANGODB_PASSWORD``. Defaults to "".

        Returns:
            An arango.database.StandardDatabase.
        """
    db = get_arangodb_client(url=url, dbname=dbname, username=username, password=password)
    return cls(db)