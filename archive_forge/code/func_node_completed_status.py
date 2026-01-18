import os
import sys
from ...interfaces.base import CommandLine
from .base import GraphPluginBase, logger
def node_completed_status(checknode):
    """
    A function to determine if a node has previously completed it's work
    :param checknode: The node to check the run status
    :return: boolean value True indicates that the node does not need to be run.
    """
    ' TODO: place this in the base.py file and refactor '
    node_state_does_not_require_overwrite = checknode.overwrite is False or (checknode.overwrite is None and (not checknode._interface.always_run))
    hash_exists = False
    try:
        hash_exists, _, _, _ = checknode.hash_exists()
    except Exception:
        hash_exists = False
    return hash_exists and node_state_does_not_require_overwrite