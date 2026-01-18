from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path
@contextlib.contextmanager
def report_compile_source_on_error():
    try:
        yield
    except Exception as exc:
        tb = exc.__traceback__
        stack = []
        while tb is not None:
            filename = tb.tb_frame.f_code.co_filename
            source = tb.tb_frame.f_globals.get('__compile_source__')
            if filename == '<string>' and source is not None:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
                    f.write(source)
                frame = tb.tb_frame
                code = compile('__inspect_currentframe()', f.name, 'eval')
                code = code.replace(co_name=frame.f_code.co_name)
                if hasattr(frame.f_code, 'co_linetable'):
                    code = code.replace(co_linetable=frame.f_code.co_linetable, co_firstlineno=frame.f_code.co_firstlineno)
                fake_frame = eval(code, frame.f_globals, {**frame.f_locals, '__inspect_currentframe': inspect.currentframe})
                fake_tb = TracebackType(None, fake_frame, tb.tb_lasti, tb.tb_lineno)
                stack.append(fake_tb)
            else:
                stack.append(tb)
            tb = tb.tb_next
        tb_next = None
        for tb in reversed(stack):
            tb.tb_next = tb_next
            tb_next = tb
        raise exc.with_traceback(tb_next)