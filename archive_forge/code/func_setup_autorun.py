import inspect
import logging
import os
import signal
import stat
import sys
import threading
import time
import traceback
from oslo_utils import timeutils
from oslo_reports.generators import conf as cgen
from oslo_reports.generators import process as prgen
from oslo_reports.generators import threading as tgen
from oslo_reports.generators import version as pgen
from oslo_reports import report
@classmethod
def setup_autorun(cls, version, service_name=None, log_dir=None, signum=None, conf=None, setup_signal=True):
    """Set Up Auto-Run

        This method sets up the Guru Meditation Report to automatically
        get dumped to stderr or a file in a given dir when the given signal
        is received. It can also use file modification events instead of
        signals.

        :param version: the version object for the current product
        :param service_name: this program name used to construct logfile name
        :param logdir: path to a log directory where to create a file
        :param signum: the signal to associate with running the report
        :param conf: Configuration object, managed by the caller.
        :param setup_signal: Set up a signal handler
        """
    if log_dir is None and conf is not None:
        log_dir = conf.oslo_reports.log_dir
    if signum:
        if setup_signal:
            cls._setup_signal(signum, version, service_name, log_dir)
        return
    if conf and conf.oslo_reports.file_event_handler:
        cls._setup_file_watcher(conf.oslo_reports.file_event_handler, conf.oslo_reports.file_event_handler_interval, version, service_name, log_dir)
    elif setup_signal and hasattr(signal, 'SIGUSR2'):
        cls._setup_signal(signal.SIGUSR2, version, service_name, log_dir)