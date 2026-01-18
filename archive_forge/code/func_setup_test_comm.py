from ipywidgets import Widget
import ipywidgets.widgets.widget
import ipykernel.comm
def setup_test_comm():
    if NEW_COMM_PACKAGE:
        comm.create_comm = dummy_create_comm
        comm.get_comm_manager = dummy_get_comm_manager
        ipykernel.comm.comm.BaseComm = DummyComm
    else:
        ipykernel.comm.Comm = DummyComm
    Widget.comm.klass = DummyComm
    ipywidgets.widgets.widget.Comm = DummyComm
    _widget_attrs['_repr_mimebundle_'] = Widget._repr_mimebundle_

    def raise_not_implemented(*args, **kwargs):
        raise NotImplementedError()
    Widget._repr_mimebundle_ = raise_not_implemented