import linecache
import sys
from IPython.core import compilerop
def test_cache_unicode():
    cp = compilerop.CachingCompiler()
    ncache = len(linecache.cache)
    cp.cache(u"t = 'žćčšđ'")
    assert len(linecache.cache) > ncache