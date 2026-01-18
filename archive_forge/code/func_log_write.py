import glob
import io
import logging
import os
import time
def log_write(self, data, kind='input'):
    """Write data to the log file, if active"""
    if self.log_active and data:
        write = self.logfile.write
        if kind == 'input':
            if self.timestamp:
                write(time.strftime('# %a, %d %b %Y %H:%M:%S\n', time.localtime()))
            write(data)
        elif kind == 'output' and self.log_output:
            odata = u'\n'.join([u'#[Out]# %s' % s for s in data.splitlines()])
            write(u'%s\n' % odata)
        try:
            self.logfile.flush()
        except OSError:
            print('Failed to flush the log file.')
            print(f'Please check that {self.logfname} exists and have the right permissions.')
            print('Also consider turning off the log with `%logstop` to avoid this warning.')