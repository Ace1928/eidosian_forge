import logging
import os
import secrets
import shutil
import tempfile
import uuid
from contextlib import suppress
from urllib.parse import quote
import requests
from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, tokenize
def get_delegation_token(self, renewer=None):
    """Retrieve token which can give the same authority to other uses

        Parameters
        ----------
        renewer: str or None
            User who may use this token; if None, will be current user
        """
    if renewer:
        out = self._call('GETDELEGATIONTOKEN', renewer=renewer)
    else:
        out = self._call('GETDELEGATIONTOKEN')
    t = out.json()['Token']
    if t is None:
        raise ValueError('No token available for this user/security context')
    return t['urlString']