from jedi import debug
from jedi.inference.base_value import ValueSet, \
from jedi.inference.utils import to_list
from jedi.inference.gradual.stub_value import StubModuleValue
from jedi.inference.gradual.typeshed import try_to_load_stub_cached
from jedi.inference.value.decorator import Decoratee
def convert_names(names, only_stubs=False, prefer_stubs=False, prefer_stub_to_compiled=True):
    if only_stubs and prefer_stubs:
        raise ValueError('You cannot use both of only_stubs and prefer_stubs.')
    with debug.increase_indent_cm('convert names'):
        if only_stubs or prefer_stubs:
            return _python_to_stub_names(names, fallback_to_python=prefer_stubs)
        else:
            return _try_stub_to_python_names(names, prefer_stub_to_compiled=prefer_stub_to_compiled)