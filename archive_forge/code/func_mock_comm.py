import pytest
from ipykernel.comm import Comm
from ipywidgets import Widget
@pytest.fixture
def mock_comm():
    _widget_attrs['_comm_default'] = getattr(Widget, '_comm_default', undefined)
    Widget._comm_default = lambda self: DummyComm()
    display_attr = '_ipython_display_' if hasattr(Widget, '_ipython_display_') else '_repr_mimebundle_'
    _widget_attrs[display_attr] = getattr(Widget, display_attr)

    def raise_not_implemented(*args, **kwargs):
        raise NotImplementedError()
    setattr(Widget, display_attr, raise_not_implemented)
    yield DummyComm()
    for attr, value in _widget_attrs.items():
        if value is undefined:
            delattr(Widget, attr)
        else:
            setattr(Widget, attr, value)