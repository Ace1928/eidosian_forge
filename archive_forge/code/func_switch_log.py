import glob
import io
import logging
import os
import time
def switch_log(self, val):
    """Switch logging on/off. val should be ONLY a boolean."""
    if val not in [False, True, 0, 1]:
        raise ValueError('Call switch_log ONLY with a boolean argument, not with: %s' % val)
    label = {0: 'OFF', 1: 'ON', False: 'OFF', True: 'ON'}
    if self.logfile is None:
        print("\nLogging hasn't been started yet (use logstart for that).\n\n%logon/%logoff are for temporarily starting and stopping logging for a logfile\nwhich already exists. But you must first start the logging process with\n%logstart (optionally giving a logfile name).")
    elif self.log_active == val:
        print('Logging is already', label[val])
    else:
        print('Switching logging', label[val])
        self.log_active = not self.log_active
        self.log_active_out = self.log_active