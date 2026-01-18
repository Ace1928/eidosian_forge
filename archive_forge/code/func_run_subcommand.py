import argparse
import logging
import os
import sys
from cliff import app
from cliff import commandmanager
from keystoneauth1 import loading
from oslo_utils import encodeutils
from blazarclient import client as blazar_client
from blazarclient import exception
from blazarclient.v1.shell_commands import allocations
from blazarclient.v1.shell_commands import floatingips
from blazarclient.v1.shell_commands import hosts
from blazarclient.v1.shell_commands import leases
from blazarclient import version as base_version
def run_subcommand(self, argv):
    subcommand = self.command_manager.find_command(argv)
    cmd_factory, cmd_name, sub_argv = subcommand
    cmd = cmd_factory(self, self.options)
    result = 1
    try:
        self.prepare_to_run_command(cmd)
        full_name = cmd_name if self.interactive_mode else ' '.join([self.NAME, cmd_name])
        cmd_parser = cmd.get_parser(full_name)
        return run_command(cmd, cmd_parser, sub_argv)
    except Exception as err:
        if self.options.debug:
            self.log.exception(str(err))
        else:
            self.log.error(str(err))
        try:
            self.clean_up(cmd, result, err)
        except Exception as err2:
            if self.options.debug:
                self.log.exception(str(err2))
            else:
                self.log.error('Could not clean up: %s', str(err2))
        if self.options.debug:
            raise
        else:
            try:
                self.clean_up(cmd, result, None)
            except Exception as err3:
                if self.options.debug:
                    self.log.exception(str(err3))
                else:
                    self.log.error('Could not clean up: %s', str(err3))
    return result