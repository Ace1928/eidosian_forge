import json
import zaqarclient.transport.errors as errors
def queue_delete(transport, request, name, callback=None):
    """Deletes queue."""
    return _common_queue_ops('queue_delete', transport, request, name, callback=callback)