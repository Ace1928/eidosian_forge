import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def new_ccompiler_opt(compiler, dispatch_hpath, **kwargs):
    """
    Create a new instance of 'CCompilerOpt' and generate the dispatch header
    which contains the #definitions and headers of platform-specific instruction-sets for
    the enabled CPU baseline and dispatch-able features.

    Parameters
    ----------
    compiler : CCompiler instance
    dispatch_hpath : str
        path of the dispatch header

    **kwargs: passed as-is to `CCompilerOpt(...)`
    Returns
    -------
    new instance of CCompilerOpt
    """
    opt = CCompilerOpt(compiler, **kwargs)
    if not os.path.exists(dispatch_hpath) or not opt.is_cached():
        opt.generate_dispatch_header(dispatch_hpath)
    return opt