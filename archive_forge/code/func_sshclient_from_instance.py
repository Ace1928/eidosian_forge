from boto.mashups.interactive import interactive_shell
import boto
import os
import time
import shutil
import paramiko
import socket
import subprocess
from boto.compat import StringIO
def sshclient_from_instance(instance, ssh_key_file, host_key_file='~/.ssh/known_hosts', user_name='root', ssh_pwd=None):
    """
    Create and return an SSHClient object given an
    instance object.

    :type instance: :class`boto.ec2.instance.Instance` object
    :param instance: The instance object.

    :type ssh_key_file: string
    :param ssh_key_file: A path to the private key file that is 
                        used to log into the instance.

    :type host_key_file: string
    :param host_key_file: A path to the known_hosts file used
                          by the SSH client.
                          Defaults to ~/.ssh/known_hosts
    :type user_name: string
    :param user_name: The username to use when logging into
                      the instance.  Defaults to root.

    :type ssh_pwd: string
    :param ssh_pwd: The passphrase, if any, associated with
                    private key.
    """
    s = FakeServer(instance, ssh_key_file)
    return SSHClient(s, host_key_file, user_name, ssh_pwd)