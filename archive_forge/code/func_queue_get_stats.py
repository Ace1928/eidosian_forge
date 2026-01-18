import json
import zaqarclient.transport.errors as errors
def queue_get_stats(transport, request, name):
    return _common_queue_ops('queue_get_stats', transport, request, name)