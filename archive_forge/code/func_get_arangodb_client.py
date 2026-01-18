import os
from math import ceil
from typing import Any, Dict, List, Optional
def get_arangodb_client(url: Optional[str]=None, dbname: Optional[str]=None, username: Optional[str]=None, password: Optional[str]=None) -> Any:
    """Get the Arango DB client from credentials.

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
    try:
        from arango import ArangoClient
    except ImportError as e:
        raise ImportError('Unable to import arango, please install with `pip install python-arango`.') from e
    _url: str = url or os.environ.get('ARANGODB_URL', 'http://localhost:8529')
    _dbname: str = dbname or os.environ.get('ARANGODB_DBNAME', '_system')
    _username: str = username or os.environ.get('ARANGODB_USERNAME', 'root')
    _password: str = password or os.environ.get('ARANGODB_PASSWORD', '')
    return ArangoClient(_url).db(_dbname, _username, _password, verify=True)