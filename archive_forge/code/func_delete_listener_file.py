import argparse
import atexit
import codecs
import locale
import os
import sys
def delete_listener_file():
    log.info('Listener ports closed; deleting {0!r}', listener_file)
    try:
        os.remove(listener_file)
    except Exception:
        log.swallow_exception('Failed to delete {0!r}', listener_file, level='warning')