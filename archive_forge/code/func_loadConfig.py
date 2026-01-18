import email.utils
import getpass
import os
import sys
from configparser import ConfigParser
from io import StringIO
from twisted.copyright import version
from twisted.internet import reactor
from twisted.logger import Logger, textFileLogObserver
from twisted.mail import smtp
def loadConfig(path):
    c = Configuration()
    if not os.access(path, os.R_OK):
        return c
    p = ConfigParser()
    p.read(path)
    au = c.allowUIDs
    du = c.denyUIDs
    ag = c.allowGIDs
    dg = c.denyGIDs
    for section, a, d in (('useraccess', au, du), ('groupaccess', ag, dg)):
        if p.has_section(section):
            for mode, L in (('allow', a), ('deny', d)):
                if p.has_option(section, mode) and p.get(section, mode):
                    for sectionID in p.get(section, mode).split(','):
                        try:
                            sectionID = int(sectionID)
                        except ValueError:
                            _log.error('Illegal {prefix}ID in [{section}] section: {sectionID}', prefix=section[0].upper(), section=section, sectionID=sectionID)
                        else:
                            L.append(sectionID)
            order = p.get(section, 'order')
            order = [s.split() for s in [s.lower() for s in order.split(',')]]
            if order[0] == 'allow':
                setattr(c, section, 'allow')
            else:
                setattr(c, section, 'deny')
    if p.has_section('identity'):
        for host, up in p.items('identity'):
            parts = up.split(':', 1)
            if len(parts) != 2:
                _log.error('Illegal entry in [identity] section: {section}', section=up)
                continue
            c.identities[host] = parts
    if p.has_section('addresses'):
        if p.has_option('addresses', 'smarthost'):
            c.smarthost = p.get('addresses', 'smarthost')
        if p.has_option('addresses', 'default_domain'):
            c.domain = p.get('addresses', 'default_domain')
    return c