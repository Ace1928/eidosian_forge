import sys
import traceback
from rdkit.rdBase import DisableLog, EnableLog, LogMessage
def logIt(self, dest, msg, *args, **kwargs):
    if args:
        msg = msg % args
    LogMessage(dest, msg + '\n')
    if kwargs.get('exc_info', False):
        exc_type, exc_val, exc_tb = sys.exc_info()
        if exc_type:
            LogMessage(dest, '\n')
            txt = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
            LogMessage(dest, txt)