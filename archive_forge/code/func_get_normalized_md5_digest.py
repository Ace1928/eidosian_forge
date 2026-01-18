from typing import Any, List
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import insecure_hash
def get_normalized_md5_digest(elements: List[Any]) -> str:
    """Computes a normalized digest for a list of hashable elements.

    Args:
        elements: A list of hashable elements for inclusion in the md5 digest.

    Returns:
        An 8-character, truncated md5 digest.
    """
    if not elements:
        raise MlflowException('No hashable elements were provided for md5 digest creation', INVALID_PARAMETER_VALUE)
    md5 = insecure_hash.md5()
    for element in elements:
        md5.update(element)
    return md5.hexdigest()[:8]