from types import FrameType, TracebackType
from typing import Callable, List, Optional
def modify_traceback(traceback: Optional[TracebackType], should_prune: Optional[Callable[[str], bool]]=None, add_traceback: Optional[TracebackType]=None) -> Optional[TracebackType]:
    ctb: Optional[TracebackType] = None
    stack: List[TracebackType] = []
    if add_traceback is not None:
        f: Optional[TracebackType] = add_traceback
        while f is not None:
            stack.append(f)
            f = f.tb_next
    f = traceback
    while f is not None:
        stack.append(f)
        f = f.tb_next
    stack.reverse()
    for n, f in enumerate(stack):
        if n == 0 or should_prune is None or (not should_prune(f.tb_frame.f_globals['__name__'])):
            ctb = TracebackType(tb_next=ctb, tb_frame=f.tb_frame, tb_lasti=f.tb_lasti, tb_lineno=f.tb_lineno)
    return ctb