import fcntl
import fnmatch
import getpass
import glob
import os
import pwd
import stat
import struct
import sys
import tty
from typing import List, Optional, TextIO, Union
from twisted.conch.client import connect, default, options
from twisted.conch.ssh import channel, common, connection, filetransfer
from twisted.internet import defer, reactor, stdio, utils
from twisted.protocols import basic
from twisted.python import failure, log, usage
from twisted.python.filepath import FilePath
def cmd_HELP(self, ignored):
    return "Available commands:\ncd path                         Change remote directory to 'path'.\nchgrp gid path                  Change gid of 'path' to 'gid'.\nchmod mode path                 Change mode of 'path' to 'mode'.\nchown uid path                  Change uid of 'path' to 'uid'.\nexit                            Disconnect from the server.\nget remote-path [local-path]    Get remote file.\nhelp                            Get a list of available commands.\nlcd path                        Change local directory to 'path'.\nlls [ls-options] [path]         Display local directory listing.\nlmkdir path                     Create local directory.\nln linkpath targetpath          Symlink remote file.\nlpwd                            Print the local working directory.\nls [-l] [path]                  Display remote directory listing.\nmkdir path                      Create remote directory.\nprogress                        Toggle progress bar.\nput local-path [remote-path]    Put local file.\npwd                             Print the remote working directory.\nquit                            Disconnect from the server.\nrename oldpath newpath          Rename remote file.\nrmdir path                      Remove remote directory.\nrm path                         Remove remote file.\nversion                         Print the SFTP version.\n?                               Synonym for 'help'.\n"