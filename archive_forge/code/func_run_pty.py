from boto.mashups.interactive import interactive_shell
import boto
import os
import time
import shutil
import paramiko
import socket
import subprocess
from boto.compat import StringIO
def run_pty(self, command):
    """
        Request a pseudo-terminal from a server, and execute a command on that
        server.

        :type command: string
        :param command: The command that you want to run on the remote host.
        
        :rtype: :class:`paramiko.channel.Channel`
        :return: An open channel object.
        """
    boto.log.debug('running:%s on %s' % (command, self.server.instance_id))
    channel = self._ssh_client.get_transport().open_session()
    channel.get_pty()
    channel.exec_command(command)
    return channel