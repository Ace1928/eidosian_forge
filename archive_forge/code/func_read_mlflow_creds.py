import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
def read_mlflow_creds() -> MlflowCreds:
    username_file, password_file = _read_mlflow_creds_from_file()
    username_env, password_env = _read_mlflow_creds_from_env()
    return MlflowCreds(username=username_env or username_file, password=password_env or password_file)