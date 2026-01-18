import os.path
from tempfile import TemporaryDirectory
import IPython.testing.tools as tt
from IPython.utils.syspathcontext import prepended_to_syspath
def test_extension_builtins():
    em = get_ipython().extension_manager
    with TemporaryDirectory() as td:
        ext3 = os.path.join(td, 'ext3.py')
        with open(ext3, 'w', encoding='utf-8') as f:
            f.write(ext3_content)
        assert 'ext3' not in em.loaded
        with prepended_to_syspath(td):
            with tt.AssertPrints('True'):
                assert em.load_extension('ext3') is None
            assert 'ext3' in em.loaded