import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
def rm_watch(self, wd, rec=False, quiet=True):
    """
        Removes watch(s).

        @param wd: Watch Descriptor of the file or directory to unwatch.
                   Also accepts a list of WDs.
        @type wd: int or list of int.
        @param rec: Recursively removes watches on every already watched
                    subdirectories and subfiles.
        @type rec: bool
        @param quiet: If False raises a WatchManagerError exception on
                      error. See example not_quiet.py
        @type quiet: bool
        @return: dict of watch descriptors associated to booleans values.
                 True if the corresponding wd has been successfully
                 removed, False otherwise.
        @rtype: dict of {int: bool}
        """
    lwd = self.__format_param(wd)
    if rec:
        lwd = self.__get_sub_rec(lwd)
    ret_ = {}
    for awd in lwd:
        wd_ = self._inotify_wrapper.inotify_rm_watch(self._fd, awd)
        if wd_ < 0:
            ret_[awd] = False
            err = 'rm_watch: cannot remove WD=%d, %s' % (awd, self._inotify_wrapper.str_errno())
            if quiet:
                log.error(err)
                continue
            raise WatchManagerError(err, ret_)
        if awd in self._wmd:
            del self._wmd[awd]
        ret_[awd] = True
        log.debug('Watch WD=%d (%s) removed', awd, self.get_path(awd))
    return ret_