from __future__ import absolute_import
from boto.mturk.test.support import unittest
def has_selenium():
    try:
        from selenium import selenium
        globals().update(selenium=selenium)
        sel = selenium(*sel_args)
        try:
            sel.do_command('shutdown', '')
        except Exception as e:
            if not 'Server Exception' in str(e):
                raise
        result = True
    except ImportError:
        result = SeleniumFailed('selenium RC not installed')
    except Exception:
        msg = 'Error occurred initializing selenium: %s' % e
        result = SeleniumFailed(msg)
    globals().update(has_selenium=lambda: result)
    return result