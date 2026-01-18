import configparser
from pathlib import Path
from typing import NamedTuple
from mlflow.environment_variables import MLFLOW_AUTH_CONFIG_PATH
def read_auth_config() -> AuthConfig:
    config_path = _get_auth_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    return AuthConfig(default_permission=config['mlflow']['default_permission'], database_uri=config['mlflow']['database_uri'], admin_username=config['mlflow']['admin_username'], admin_password=config['mlflow']['admin_password'], authorization_function=config['mlflow'].get('authorization_function', 'mlflow.server.auth:authenticate_request_basic_auth'))