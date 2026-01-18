from typing import Tuple
from azure.core.exceptions import HttpResponseError  # type: ignore
from azure.identity import DefaultAzureCredential  # type: ignore
from azure.storage.blob import BlobClient, BlobServiceClient  # type: ignore
from ..errors import LaunchError
from ..utils import AZURE_BLOB_REGEX
from .abstract import AbstractEnvironment
@classmethod
def get_credentials(cls) -> DefaultAzureCredential:
    """Get Azure credentials."""
    try:
        return DefaultAzureCredential()
    except Exception as e:
        raise LaunchError(f'Could not get Azure credentials. Please make sure you have configured your Azure CLI correctly.\n{e}') from e