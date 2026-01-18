import os
from kubernetes.client import Configuration
from .config_exception import ConfigException
def load_incluster_config():
    """
    Use the service account kubernetes gives to pods to connect to kubernetes
    cluster. It's intended for clients that expect to be running inside a pod
    running on kubernetes. It will raise an exception if called from a process
    not running in a kubernetes environment.
  """
    InClusterConfigLoader(token_filename=SERVICE_TOKEN_FILENAME, cert_filename=SERVICE_CERT_FILENAME).load_and_set()