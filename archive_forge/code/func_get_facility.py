from __future__ import absolute_import, division, print_function
import syslog
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def get_facility(facility):
    return {'kern': syslog.LOG_KERN, 'user': syslog.LOG_USER, 'mail': syslog.LOG_MAIL, 'daemon': syslog.LOG_DAEMON, 'auth': syslog.LOG_AUTH, 'lpr': syslog.LOG_LPR, 'news': syslog.LOG_NEWS, 'uucp': syslog.LOG_UUCP, 'cron': syslog.LOG_CRON, 'syslog': syslog.LOG_SYSLOG, 'local0': syslog.LOG_LOCAL0, 'local1': syslog.LOG_LOCAL1, 'local2': syslog.LOG_LOCAL2, 'local3': syslog.LOG_LOCAL3, 'local4': syslog.LOG_LOCAL4, 'local5': syslog.LOG_LOCAL5, 'local6': syslog.LOG_LOCAL6, 'local7': syslog.LOG_LOCAL7}.get(facility, syslog.LOG_DAEMON)