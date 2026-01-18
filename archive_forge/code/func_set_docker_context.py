import json
import os
import sys
def set_docker_context(path):
    """Send updated Docker context to the controller.

    Args:
        path: (str) new directory to use as docker context.
    """
    _write_msg(type='set_docker_context', path=path)