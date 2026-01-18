import warnings
from abc import ABCMeta, abstractmethod
import entrypoints
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.uri import get_uri_scheme
def get_store_builder(self, store_uri):
    """Get a store from the registry based on the scheme of store_uri

        Args:
            store_uri: The store URI. If None, it will be inferred from the environment. This
                URI is used to select which tracking store implementation to instantiate
                and is passed to the constructor of the implementation.

        Returns:
            A function that returns an instance of
            ``mlflow.store.{tracking|model_registry}.AbstractStore`` that fulfills the store
            URI requirements.
        """
    scheme = store_uri if store_uri in {'databricks', 'databricks-uc'} else get_uri_scheme(store_uri)
    try:
        store_builder = self._registry[scheme]
    except KeyError:
        raise UnsupportedModelRegistryStoreURIException(unsupported_uri=store_uri, supported_uri_schemes=list(self._registry.keys()))
    return store_builder