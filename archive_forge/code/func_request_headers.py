from abc import ABCMeta, abstractmethod
from mlflow.utils.annotations import developer_stable
@abstractmethod
def request_headers(self):
    """Generate context-specific request headers.

        Returns:
            dict of request headers.
        """
    pass