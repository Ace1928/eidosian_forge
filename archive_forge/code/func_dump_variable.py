import os
from distutils.dist import Distribution
def dump_variable(self, name):
    conf_desc = self._conf_keys[name]
    hook, envvar, confvar, convert, append = conf_desc
    if not convert:
        convert = lambda x: x
    print('%s.%s:' % (self._distutils_section, name))
    v = self._hook_handler(name, hook)
    print('  hook   : %s' % (convert(v),))
    if envvar:
        v = os.environ.get(envvar, None)
        print('  environ: %s' % (convert(v),))
    if confvar and self._conf:
        v = self._conf.get(confvar, (None, None))[1]
        print('  config : %s' % (convert(v),))