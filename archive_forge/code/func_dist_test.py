import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def dist_test(self, source, flags, macros=[]):
    """Return True if 'CCompiler.compile()' able to compile
        a source file with certain flags.
        """
    assert isinstance(source, str)
    from distutils.errors import CompileError
    cc = self._ccompiler
    bk_spawn = getattr(cc, 'spawn', None)
    if bk_spawn:
        cc_type = getattr(self._ccompiler, 'compiler_type', '')
        if cc_type in ('msvc',):
            setattr(cc, 'spawn', self._dist_test_spawn_paths)
        else:
            setattr(cc, 'spawn', self._dist_test_spawn)
    test = False
    try:
        self.dist_compile([source], flags, macros=macros, output_dir=self.conf_tmp_path)
        test = True
    except CompileError as e:
        self.dist_log(str(e), stderr=True)
    if bk_spawn:
        setattr(cc, 'spawn', bk_spawn)
    return test