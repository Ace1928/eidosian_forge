import datetime
import errno
import os
import sys
def handle_part(data, ctype, filename, payload):
    if ctype == '__begin__':
        try:
            os.makedirs('/var/lib/heat-cfntools', int('700', 8))
        except OSError:
            ex_type, e, tb = sys.exc_info()
            if e.errno != errno.EEXIST:
                raise
        return
    if ctype == '__end__':
        return
    timestamp = datetime.datetime.now()
    with open('/var/log/part-handler.log', 'a') as log:
        log.write('%s filename:%s, ctype:%s\n' % (timestamp, filename, ctype))
    if ctype == 'text/x-cfninitdata':
        with open('/var/lib/heat-cfntools/%s' % filename, 'w') as f:
            f.write(payload)
        with open('/var/lib/cloud/data/%s' % filename, 'w') as f:
            f.write(payload)