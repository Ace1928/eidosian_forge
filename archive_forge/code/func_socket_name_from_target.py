import sys
import ovs.util
def socket_name_from_target(target):
    assert isinstance(target, str)
    " On Windows an absolute path contains ':' ( i.e: C:\\ ) "
    if target.startswith('/') or target.find(':') > -1:
        return (0, target)
    pidfile_name = '%s/%s.pid' % (ovs.dirs.RUNDIR, target)
    pid = ovs.daemon.read_pidfile(pidfile_name)
    if pid < 0:
        return (-pid, 'cannot read pidfile "%s"' % pidfile_name)
    if sys.platform == 'win32':
        return (0, '%s/%s.ctl' % (ovs.dirs.RUNDIR, target))
    else:
        return (0, '%s/%s.%d.ctl' % (ovs.dirs.RUNDIR, target, pid))