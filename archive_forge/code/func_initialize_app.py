import argparse
import logging
import os
import sys
from cliff import app
from cliff import commandmanager
from osc_lib.command import command
from mistralclient.api import client
from mistralclient.auth import auth_types
import mistralclient.commands.v2.action_executions
import mistralclient.commands.v2.actions
import mistralclient.commands.v2.code_sources
import mistralclient.commands.v2.cron_triggers
import mistralclient.commands.v2.dynamic_actions
import mistralclient.commands.v2.environments
import mistralclient.commands.v2.event_triggers
import mistralclient.commands.v2.executions
import mistralclient.commands.v2.members
import mistralclient.commands.v2.services
import mistralclient.commands.v2.tasks
import mistralclient.commands.v2.workbooks
import mistralclient.commands.v2.workflows
from mistralclient import exceptions as exe
def initialize_app(self, argv):
    self._clear_shell_commands()
    ver = client.determine_client_version(self.options.mistral_version)
    self._set_shell_commands(self._get_commands(ver))
    need_client = not ('bash-completion' in argv or 'help' in argv or '-h' in argv or ('--help' in argv))
    if not self.options.auth_url:
        if self.options.password or self.options.token:
            self.options.auth_url = 'http://localhost:35357/v3'
    if self.options.auth_type == 'keystone' and (not self.options.auth_url.endswith('/v2.0')):
        if not self.options.project_domain_id and (not self.options.project_domain_name):
            self.options.project_domain_id = 'default'
        if not self.options.user_domain_id and (not self.options.user_domain_name):
            self.options.user_domain_id = 'default'
        if not self.options.target_project_domain_id and (not self.options.target_project_domain_name):
            self.options.target_project_domain_id = 'default'
        if not self.options.target_user_domain_id and (not self.options.target_user_domain_name):
            self.options.target_user_domain_id = 'default'
    if self.options.auth_url and (not self.options.token):
        if not self.options.username:
            raise exe.IllegalArgumentException('You must provide a username via --os-username env[OS_USERNAME]')
        if not self.options.password:
            raise exe.IllegalArgumentException('You must provide a password via --os-password env[OS_PASSWORD]')
    self.client = self._create_client() if need_client else None
    ClientManager = type('ClientManager', (object,), dict(workflow_engine=self.client))
    self.client_manager = ClientManager()