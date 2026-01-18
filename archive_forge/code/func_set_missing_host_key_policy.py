from binascii import hexlify
import getpass
import inspect
import os
import socket
import warnings
from errno import ECONNREFUSED, EHOSTUNREACH
from paramiko.agent import Agent
from paramiko.common import DEBUG
from paramiko.config import SSH_PORT
from paramiko.dsskey import DSSKey
from paramiko.ecdsakey import ECDSAKey
from paramiko.ed25519key import Ed25519Key
from paramiko.hostkeys import HostKeys
from paramiko.rsakey import RSAKey
from paramiko.ssh_exception import (
from paramiko.transport import Transport
from paramiko.util import ClosingContextManager
def set_missing_host_key_policy(self, policy):
    """
        Set policy to use when connecting to servers without a known host key.

        Specifically:

        * A **policy** is a "policy class" (or instance thereof), namely some
          subclass of `.MissingHostKeyPolicy` such as `.RejectPolicy` (the
          default), `.AutoAddPolicy`, `.WarningPolicy`, or a user-created
          subclass.
        * A host key is **known** when it appears in the client object's cached
          host keys structures (those manipulated by `load_system_host_keys`
          and/or `load_host_keys`).

        :param .MissingHostKeyPolicy policy:
            the policy to use when receiving a host key from a
            previously-unknown server
        """
    if inspect.isclass(policy):
        policy = policy()
    self._policy = policy