import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
def collectLine(self, tb, extra_data):
    f = tb.tb_frame
    lineno = tb.tb_lineno
    co = f.f_code
    filename = co.co_filename
    name = co.co_name
    globals = f.f_globals
    locals = f.f_locals
    if not hasattr(locals, 'keys'):
        warnings.warn('Frame %s has an invalid locals(): %r' % (globals.get('__name__', 'unknown'), locals))
        locals = {}
    data = {}
    data['modname'] = globals.get('__name__', None)
    data['filename'] = filename
    data['lineno'] = lineno
    data['revision'] = self.getRevision(globals)
    data['name'] = name
    data['tbid'] = id(tb)
    if '__traceback_supplement__' in locals:
        tbs = locals['__traceback_supplement__']
    elif '__traceback_supplement__' in globals:
        tbs = globals['__traceback_supplement__']
    else:
        tbs = None
    if tbs is not None:
        factory = tbs[0]
        args = tbs[1:]
        try:
            supp = factory(*args)
            data['supplement'] = self.collectSupplement(supp, tb)
            if data['supplement'].extra:
                for key, value in data['supplement'].extra.items():
                    extra_data.setdefault(key, []).append(value)
        except:
            if DEBUG_EXCEPTION_FORMATTER:
                out = StringIO()
                traceback.print_exc(file=out)
                text = out.getvalue()
                data['supplement_exception'] = text
    try:
        tbi = locals.get('__traceback_info__', None)
        if tbi is not None:
            data['traceback_info'] = str(tbi)
    except:
        pass
    marker = []
    for name in ('__traceback_hide__', '__traceback_log__', '__traceback_decorator__'):
        try:
            tbh = locals.get(name, globals.get(name, marker))
            if tbh is not marker:
                data[name[2:-2]] = tbh
        except:
            pass
    return data