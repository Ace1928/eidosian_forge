from concurrent import futures
import enum
import errno
import io
import logging as pylogging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import eventlet
from eventlet import patcher
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_privsep._i18n import _
from oslo_privsep import capabilities
from oslo_privsep import comm
def helper_main():
    """Start privileged process, serving requests over a Unix socket."""
    cfg.CONF.register_cli_opts([cfg.StrOpt('privsep_context', required=True), cfg.StrOpt('privsep_sock_path', required=True)])
    logging.register_options(cfg.CONF)
    cfg.CONF(args=sys.argv[1:], project='privsep')
    try:
        logging.setup(cfg.CONF, 'privsep', fix_eventlet=False)
    except TypeError:
        logging.setup(cfg.CONF, 'privsep')
    context = importutils.import_class(cfg.CONF.privsep_context)
    from oslo_privsep import priv_context
    if not isinstance(context, priv_context.PrivContext):
        LOG.fatal('--privsep_context must be the (python) name of a PrivContext object')
    sock = socket.socket(socket.AF_UNIX)
    sock.connect(cfg.CONF.privsep_sock_path)
    set_cloexec(sock)
    channel = comm.ServerChannel(sock)
    if os.fork() != 0:
        return
    replace_logging(PrivsepLogHandler(channel))
    LOG.info('privsep daemon starting')
    try:
        Daemon(channel, context).run()
    except Exception as e:
        LOG.exception(e)
        sys.exit(str(e))
    LOG.debug('privsep daemon exiting')
    sys.exit(0)