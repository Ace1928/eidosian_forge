import json
import posixpath
from typing import Any, Dict, Iterator, Optional
from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments.constants import (
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils import AttrDict
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
@experimental
def predict_stream(self, deployment_name=None, inputs=None, endpoint=None) -> Iterator[Dict[str, Any]]:
    """
        Submit a query to a configured provider endpoint, and get streaming response

        Args:
            deployment_name: Unused.
            inputs: The inputs to the query, as a dictionary.
            endpoint: The name of the endpoint to query.

        Returns:
            An iterator of dictionary containing the response from the endpoint.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("databricks")
            chunk_iter = client.predict_stream(
                endpoint="databricks-llama-2-70b-chat",
                inputs={
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "temperature": 0.0,
                    "n": 1,
                    "max_tokens": 500,
                },
            )
            for chunk in chunk_iter:
                print(chunk)
                # Example:
                # {
                #     "id": "82a834f5-089d-4fc0-ad6c-db5c7d6a6129",
                #     "object": "chat.completion.chunk",
                #     "created": 1712133837,
                #     "model": "llama-2-70b-chat-030424",
                #     "choices": [
                #         {
                #             "index": 0, "delta": {"role": "assistant", "content": "Hello"},
                #             "finish_reason": None,
                #         }
                #     ],
                #     "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
                # }
        """
    inputs = inputs or {}
    chunk_line_iter = self._call_endpoint_stream(method='POST', prefix='/', route=posixpath.join(endpoint, 'invocations'), json_body={**inputs, 'stream': True}, timeout=MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT.get())
    for line in chunk_line_iter:
        splits = line.split(':', 1)
        if len(splits) < 2:
            raise MlflowException(f"Unknown streaming response format: '{line}'.")
        key, value = splits
        if key != 'data':
            raise MlflowException(f"Unknown streaming response format with key '{key}'.")
        value = value.strip()
        if value == '[DONE]':
            return
        yield json.loads(value)