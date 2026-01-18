import configparser
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.i18n import _, _LE
def is_multiple_swift_store_accounts_enabled():
    if CONF.swift_store_config_file is None:
        return False
    return True