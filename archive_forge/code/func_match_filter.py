import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
def match_filter(filter_list, userargs, exec_dirs=None):
    """Checks user command and arguments through command filters.

    Returns the first matching filter.

    Raises NoFilterMatched if no filter matched.
    Raises FilterMatchNotExecutable if no executable was found for the
    best filter match.
    """
    first_not_executable_filter = None
    exec_dirs = exec_dirs or []
    for f in filter_list:
        if f.match(userargs):
            if isinstance(f, filters.ChainingFilter):

                def non_chain_filter(fltr):
                    return fltr.run_as == f.run_as and (not isinstance(fltr, filters.ChainingFilter))
                leaf_filters = [fltr for fltr in filter_list if non_chain_filter(fltr)]
                args = f.exec_args(userargs)
                if not args:
                    continue
                try:
                    match_filter(leaf_filters, args, exec_dirs=exec_dirs)
                except (NoFilterMatched, FilterMatchNotExecutable):
                    continue
            if not f.get_exec(exec_dirs=exec_dirs):
                if not first_not_executable_filter:
                    first_not_executable_filter = f
                continue
            return f
    if first_not_executable_filter:
        raise FilterMatchNotExecutable(match=first_not_executable_filter)
    raise NoFilterMatched()