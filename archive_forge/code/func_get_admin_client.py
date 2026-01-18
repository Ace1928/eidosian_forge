import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
@classmethod
def get_admin_client(cls):
    admin_client = base.CLIClient(username=CONF.admin_username, password=CONF.admin_password, tenant_name=CONF.admin_tenant_name, uri=CONF.admin_auth_url, cli_dir=CONF.manila_exec_dir, insecure=CONF.insecure, project_domain_name=CONF.admin_project_domain_name or None, project_domain_id=CONF.admin_project_domain_id or None, user_domain_name=CONF.admin_user_domain_name or None, user_domain_id=CONF.admin_user_domain_id or None)
    return admin_client