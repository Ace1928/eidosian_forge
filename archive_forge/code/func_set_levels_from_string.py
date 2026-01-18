import datetime
import logging
import logging.handlers
import os
import re
import socket
import sys
import threading
import ovs.dirs
import ovs.unixctl
import ovs.util
@staticmethod
def set_levels_from_string(s):
    module = None
    level = None
    destination = None
    words = re.split('[ :]', s)
    if words[0] == 'pattern':
        try:
            if words[1] in DESTINATIONS and words[2]:
                segments = [words[i] for i in range(2, len(words))]
                pattern = ''.join(segments)
                Vlog.set_pattern(words[1], pattern)
                return
            else:
                return 'Destination %s does not exist' % words[1]
        except IndexError:
            return 'Please supply a valid pattern and destination'
    elif words[0] == 'FACILITY':
        if words[1] in FACILITIES:
            try:
                Vlog.add_syslog_handler(words[1])
            except (IOError, socket.error):
                logger = logging.getLogger('syslog')
                logger.disabled = True
            return
        else:
            return 'Facility %s is invalid' % words[1]
    for word in [w.lower() for w in words]:
        if word == 'any':
            pass
        elif word in DESTINATIONS:
            if destination:
                return 'cannot specify multiple destinations'
            destination = word
        elif word in LEVELS:
            if level:
                return 'cannot specify multiple levels'
            level = word
        elif word in Vlog.__mfl:
            if module:
                return 'cannot specify multiple modules'
            module = word
        else:
            return 'no destination, level, or module "%s"' % word
    Vlog.set_level(module or 'any', destination or 'any', level or 'any')