import atexit
import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional
from .utils import experimental, is_gradio_available
from .utils._deprecation import _deprecate_method
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
def add_webhook(self, path: Optional[str]=None) -> Callable:
    """
        Decorator to add a webhook to the [`WebhooksServer`] server.

        Args:
            path (`str`, optional):
                The URL path to register the webhook function. If not provided, the function name will be used as the
                path. In any case, all webhooks are registered under `/webhooks`.

        Raises:
            ValueError: If the provided path is already registered as a webhook.

        Example:
            ```python
            from huggingface_hub import WebhooksServer, WebhookPayload

            app = WebhooksServer()

            @app.add_webhook
            async def trigger_training(payload: WebhookPayload):
                if payload.repo.type == "dataset" and payload.event.action == "update":
                    # Trigger a training job if a dataset is updated
                    ...

            app.run()
        ```
        """
    if callable(path):
        return self.add_webhook()(path)

    @wraps(FastAPI.post)
    def _inner_post(*args, **kwargs):
        func = args[0]
        abs_path = f'/webhooks/{(path or func.__name__).strip('/')}'
        if abs_path in self.registered_webhooks:
            raise ValueError(f'Webhook {abs_path} already exists.')
        self.registered_webhooks[abs_path] = func
    return _inner_post