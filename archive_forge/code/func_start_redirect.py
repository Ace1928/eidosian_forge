from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
def start_redirect(keep_original_redirection=False, std='stdout', redirect_to=None):
    """
    @param std: 'stdout', 'stderr', or 'both'
    """
    with _RedirectionsHolder._lock:
        if redirect_to is None:
            redirect_to = IOBuf()
        if std == 'both':
            config_stds = ['stdout', 'stderr']
        else:
            config_stds = [std]
        for std in config_stds:
            original = getattr(sys, std)
            stack = getattr(_RedirectionsHolder, '_stack_%s' % std)
            if keep_original_redirection:
                wrap_buffer = True if hasattr(redirect_to, 'buffer') else False
                new_std_instance = IORedirector(getattr(sys, std), redirect_to, wrap_buffer=wrap_buffer)
                setattr(sys, std, new_std_instance)
            else:
                new_std_instance = redirect_to
                setattr(sys, std, redirect_to)
            stack.append(_RedirectInfo(original, new_std_instance))
        return redirect_to