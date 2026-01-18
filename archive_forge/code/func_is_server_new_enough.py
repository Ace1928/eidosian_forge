import enum
import functools
import redis
from redis import exceptions as redis_exceptions
def is_server_new_enough(client, min_version, default=False, prior_version=None):
    """Checks if a client is attached to a new enough redis server."""
    if not prior_version:
        try:
            server_info = client.info()
        except redis_exceptions.ResponseError:
            server_info = {}
        version_text = server_info.get('redis_version', '')
    else:
        version_text = prior_version
    version_pieces = []
    for p in version_text.split('.'):
        try:
            version_pieces.append(int(p))
        except ValueError:
            break
    if not version_pieces:
        return (default, version_text)
    else:
        version_pieces = tuple(version_pieces)
        return (version_pieces >= min_version, version_text)