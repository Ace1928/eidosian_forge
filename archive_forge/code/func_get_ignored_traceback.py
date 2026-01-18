import functools
import traceback
def get_ignored_traceback(tb):
    """Retrieve the first traceback of an ignored trailing chain.

    Given an initial traceback, find the first traceback of a trailing
    chain of tracebacks that should be ignored.  The criteria for
    whether a traceback should be ignored is whether its frame's
    globals include the __unittest marker variable. This criteria is
    culled from:

        unittest.TestResult._is_relevant_tb_level

    For example:

       tb.tb_next => tb0.tb_next => tb1.tb_next

    - If no tracebacks were to be ignored, None would be returned.
    - If only tb1 was to be ignored, tb1 would be returned.
    - If tb0 and tb1 were to be ignored, tb0 would be returned.
    - If either of only tb or only tb0 was to be ignored, None would
      be returned because neither tb or tb0 would be part of a
      trailing chain of ignored tracebacks.
    """
    tb_list = []
    while tb:
        tb_list.append(tb)
        tb = tb.tb_next
    ignored_tracebacks = []
    for tb in reversed(tb_list):
        if '__unittest' in tb.tb_frame.f_globals:
            ignored_tracebacks.append(tb)
        else:
            break
    if ignored_tracebacks:
        return ignored_tracebacks[-1]