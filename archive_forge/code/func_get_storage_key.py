import abc
from abc import abstractmethod
from typing import Optional
from ray.util.annotations import DeveloperAPI
@abstractmethod
def get_storage_key(self, key: str) -> str:
    """Get internal key for storage.

        Args:
            key: User provided key

        Returns:
            storage_key: Formatted key for storage, usually by
                prepending namespace.
        """
    raise NotImplementedError('get_storage_key() has to be implemented')