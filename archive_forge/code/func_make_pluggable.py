import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@classmethod
def make_pluggable(cls, audience=AUDIENCE, subject_token_type=SUBJECT_TOKEN_TYPE, token_url=TOKEN_URL, token_info_url=TOKEN_INFO_URL, client_id=None, client_secret=None, quota_project_id=None, scopes=None, default_scopes=None, service_account_impersonation_url=None, credential_source=None, workforce_pool_user_project=None, interactive=None):
    return pluggable.Credentials(audience=audience, subject_token_type=subject_token_type, token_url=token_url, token_info_url=token_info_url, service_account_impersonation_url=service_account_impersonation_url, credential_source=credential_source, client_id=client_id, client_secret=client_secret, quota_project_id=quota_project_id, scopes=scopes, default_scopes=default_scopes, workforce_pool_user_project=workforce_pool_user_project, interactive=interactive)