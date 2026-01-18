from __future__ import annotations
import os
from typing import Any, Collection, cast
def load_from_snowsql_config_file(connection_name: str) -> dict[str, Any]:
    """Loads the dictionary from snowsql config file."""
    snowsql_config_file = os.path.expanduser(SNOWSQL_CONNECTION_FILE)
    if not os.path.exists(snowsql_config_file):
        return {}
    import configparser
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(snowsql_config_file)
    if f'connections.{connection_name}' in config:
        raw_conn_params = config[f'connections.{connection_name}']
    elif 'connections' in config:
        raw_conn_params = config['connections']
    else:
        return {}
    conn_params = {k.replace('name', ''): v.strip('"') for k, v in raw_conn_params.items()}
    if 'db' in conn_params:
        conn_params['database'] = conn_params['db']
        del conn_params['db']
    return conn_params