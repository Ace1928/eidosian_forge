import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def poll_for_status(poll_fn, obj_id, action, final_ok_states, poll_period=5, show_progress=True):
    """Blocks while an action occurs. Periodically shows progress."""

    def print_progress(progress):
        if show_progress:
            msg = '\rInstance %(action)s... %(progress)s%% complete' % dict(action=action, progress=progress)
        else:
            msg = '\rInstance %(action)s...' % dict(action=action)
        sys.stdout.write(msg)
        sys.stdout.flush()
    print()
    while True:
        obj = poll_fn(obj_id)
        status = obj.status.lower()
        progress = getattr(obj, 'progress', None) or 0
        if status in final_ok_states:
            print_progress(100)
            print('\nFinished')
            break
        elif status == 'error':
            print('\nError %(action)s instance' % {'action': action})
            break
        else:
            print_progress(progress)
            time.sleep(poll_period)